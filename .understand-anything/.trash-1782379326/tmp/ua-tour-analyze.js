#!/usr/bin/env node
'use strict';

const fs = require('fs');

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error('Usage: ua-tour-analyze.js <input.json> <output.json>');
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  const nodes = raw.nodes || [];
  const edges = raw.edges || [];
  const layers = raw.layers || [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);

  // Only consider edges whose endpoints are both file-level nodes we kept
  const fileEdges = edges.filter(e => nodeById.has(e.source) && nodeById.has(e.target));

  // ---- Fan-in / Fan-out ----
  const fanIn = new Map();
  const fanOut = new Map();
  for (const n of nodes) { fanIn.set(n.id, 0); fanOut.set(n.id, 0); }
  for (const e of fileEdges) {
    fanOut.set(e.source, (fanOut.get(e.source) || 0) + 1);
    fanIn.set(e.target, (fanIn.get(e.target) || 0) + 1);
  }

  const nameOf = id => (nodeById.get(id) ? nodeById.get(id).name : id);
  const summaryOf = id => (nodeById.get(id) ? (nodeById.get(id).summary || '') : '');

  const fanInRanking = [...fanIn.entries()]
    .map(([id, v]) => ({ id, fanIn: v, name: nameOf(id) }))
    .sort((a, b) => b.fanIn - a.fanIn)
    .slice(0, 20);

  const fanOutRanking = [...fanOut.entries()]
    .map(([id, v]) => ({ id, fanOut: v, name: nameOf(id) }))
    .sort((a, b) => b.fanOut - a.fanOut)
    .slice(0, 20);

  // ---- Entry point candidates ----
  const codeEntryNames = new Set([
    'index.ts','index.js','main.ts','main.js','app.ts','app.js','server.ts','server.js',
    'mod.rs','main.go','main.py','main.rs','manage.py','app.py','wsgi.py','asgi.py','run.py',
    '__main__.py','Application.java','Main.java','Program.cs','config.ru','index.php',
    'App.swift','Application.kt','main.cpp','main.c'
  ]);

  const fanOutValues = [...fanOut.values()].sort((a, b) => b - a);
  const top10pctIdx = Math.max(0, Math.floor(fanOutValues.length * 0.10) - 1);
  const top10pctThreshold = fanOutValues.length ? fanOutValues[top10pctIdx] : Infinity;
  const fanInValues = [...fanIn.values()].sort((a, b) => a - b);
  const bottom25pctIdx = Math.max(0, Math.floor(fanInValues.length * 0.25) - 1);
  const bottom25pctThreshold = fanInValues.length ? fanInValues[bottom25pctIdx] : 0;

  const entryScores = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || '';
    const depth = fp.split('/').length;
    if (n.type === 'document') {
      if (n.name === 'README.md' && depth === 1) score += 5;
      else if (/\.md$/i.test(n.name) && depth === 1) score += 2;
    } else if (n.type === 'file') {
      if (codeEntryNames.has(n.name)) score += 3;
      if (depth <= 2) score += 1;
      if ((fanOut.get(n.id) || 0) >= top10pctThreshold && (fanOut.get(n.id) || 0) > 0) score += 1;
      if ((fanIn.get(n.id) || 0) <= bottom25pctThreshold) score += 1;
    }
    if (score > 0) entryScores.push({ id: n.id, score, name: n.name, summary: summaryOf(n.id) });
  }
  entryScores.sort((a, b) => b.score - a.score);
  const entryPointCandidates = entryScores.slice(0, 5);

  // ---- BFS from top CODE entry point ----
  const traversalEdgeTypes = new Set(['imports', 'calls']);
  const adj = new Map();
  for (const n of nodes) adj.set(n.id, []);
  for (const e of fileEdges) {
    if (traversalEdgeTypes.has(e.type)) adj.get(e.source).push(e.target);
  }

  // top code entry: prefer app/main.py, else first non-document candidate
  let startNode = null;
  const mainPy = nodes.find(n => n.filePath === 'app/main.py');
  if (mainPy) startNode = mainPy.id;
  if (!startNode) {
    const codeCand = entryPointCandidates.find(c => {
      const nn = nodeById.get(c.id);
      return nn && nn.type !== 'document';
    });
    if (codeCand) startNode = codeCand.id;
  }
  if (!startNode && nodes.length) startNode = nodes[0].id;

  const order = [];
  const depthMap = {};
  const byDepth = {};
  if (startNode) {
    const visited = new Set([startNode]);
    const queue = [[startNode, 0]];
    while (queue.length) {
      const [cur, d] = queue.shift();
      order.push(cur);
      depthMap[cur] = d;
      (byDepth[d] = byDepth[d] || []).push(cur);
      for (const nb of (adj.get(cur) || [])) {
        if (!visited.has(nb)) { visited.add(nb); queue.push([nb, d + 1]); }
      }
    }
  }

  // ---- Non-code inventory ----
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const item = { id: n.id, name: n.name, summary: n.summary || '' };
    if (n.type === 'document') nonCodeFiles.documentation.push(item);
    else if (['service', 'pipeline', 'resource'].includes(n.type)) nonCodeFiles.infrastructure.push(item);
    else if (['table', 'schema', 'endpoint'].includes(n.type)) nonCodeFiles.data.push(item);
    else if (n.type === 'config') nonCodeFiles.config.push(item);
  }

  // ---- Clusters: bidirectional pairs expanded ----
  const pairKey = (a, b) => (a < b ? a + '||' + b : b + '||' + a);
  const directed = new Set();
  for (const e of fileEdges) {
    if (traversalEdgeTypes.has(e.type)) directed.add(e.source + '>>' + e.target);
  }
  const edgeCountBetween = new Map(); // undirected count
  for (const e of fileEdges) {
    if (!traversalEdgeTypes.has(e.type)) continue;
    const k = pairKey(e.source, e.target);
    edgeCountBetween.set(k, (edgeCountBetween.get(k) || 0) + 1);
  }
  // seed clusters from bidirectional pairs
  const clusters = [];
  const seen = new Set();
  for (const e of fileEdges) {
    if (!traversalEdgeTypes.has(e.type)) continue;
    const a = e.source, b = e.target;
    if (directed.has(b + '>>' + a)) {
      const k = pairKey(a, b);
      if (seen.has(k)) continue;
      seen.add(k);
      const members = new Set([a, b]);
      // expand: add nodes connected to >=2 members
      for (const n of nodes) {
        if (members.has(n.id)) continue;
        let conn = 0;
        for (const m of members) {
          if (directed.has(n.id + '>>' + m) || directed.has(m + '>>' + n.id)) conn++;
        }
        if (conn >= 2 && members.size < 5) members.add(n.id);
      }
      let ec = 0;
      const arr = [...members];
      for (let i = 0; i < arr.length; i++)
        for (let j = i + 1; j < arr.length; j++)
          ec += edgeCountBetween.get(pairKey(arr[i], arr[j])) || 0;
      clusters.push({ nodes: arr, edgeCount: ec });
    }
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount);
  const topClusters = clusters.slice(0, 10);

  // ---- Node summary index ----
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary || '' };
  }

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode, order, depthMap, byDepth },
    nonCodeFiles,
    clusters: topClusters,
    layers: { count: layers.length, list: layers },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: fileEdges.length
  };

  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

try { main(); } catch (err) { console.error(err && err.stack ? err.stack : String(err)); process.exit(1); }
