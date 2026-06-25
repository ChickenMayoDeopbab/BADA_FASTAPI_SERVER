#!/usr/bin/env node
"use strict";
const fs = require("fs");

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  console.error("Usage: node ua-arch-analyze.js <input.json> <output.json>");
  process.exit(1);
}

let data;
try {
  data = JSON.parse(fs.readFileSync(inputPath, "utf8"));
} catch (e) {
  console.error("Failed to read/parse input: " + e.message);
  process.exit(1);
}

const fileNodes = data.fileNodes || [];
const importEdges = data.importEdges || [];
const allEdges = data.allEdges || [];

const byId = {};
for (const n of fileNodes) byId[n.id] = n;

// ---- Common prefix computation ----
function dirOf(p) {
  const idx = p.lastIndexOf("/");
  return idx === -1 ? "" : p.slice(0, idx);
}
const paths = fileNodes.map((n) => n.filePath || "");
// common directory prefix across all paths
function commonPrefix(arr) {
  if (arr.length === 0) return "";
  const split = arr.map((p) => p.split("/"));
  const first = split[0];
  let prefix = [];
  for (let i = 0; i < first.length - 0; i++) {
    const seg = first[i];
    if (split.every((s) => s.length > i + 1 && s[i] === seg)) {
      prefix.push(seg);
    } else break;
  }
  return prefix.length ? prefix.join("/") + "/" : "";
}
const prefix = commonPrefix(paths);

function groupKey(p) {
  let rest = p;
  if (prefix && rest.startsWith(prefix)) rest = rest.slice(prefix.length);
  const segs = rest.split("/");
  if (segs.length <= 1) return "(root)";
  return segs[0];
}

// ---- A. Directory grouping ----
const directoryGroups = {};
for (const n of fileNodes) {
  const g = groupKey(n.filePath || "");
  (directoryGroups[g] = directoryGroups[g] || []).push(n.id);
}

// ---- B. Node type grouping ----
const nodeTypeGroups = {};
for (const n of fileNodes) {
  const t = n.type || "file";
  (nodeTypeGroups[t] = nodeTypeGroups[t] || []).push(n.id);
}

// ---- Group lookup for an id ----
const idToGroup = {};
for (const g in directoryGroups) for (const id of directoryGroups[g]) idToGroup[id] = g;

// ---- C. Fan-in/out ----
const fanOut = {};
const fanIn = {};
for (const n of fileNodes) {
  fanOut[n.id] = 0;
  fanIn[n.id] = 0;
}
for (const e of importEdges) {
  if (fanOut[e.source] !== undefined) fanOut[e.source]++;
  if (fanIn[e.target] !== undefined) fanIn[e.target]++;
}

// ---- D. Cross-category edges ----
const crossMap = {};
for (const e of allEdges) {
  const s = byId[e.source], t = byId[e.target];
  if (!s || !t) continue;
  if (s.type === t.type) continue;
  const key = s.type + "|" + t.type + "|" + (e.type || "");
  crossMap[key] = (crossMap[key] || 0) + 1;
}
const crossCategoryEdges = Object.keys(crossMap).map((k) => {
  const [fromType, toType, edgeType] = k.split("|");
  return { fromType, toType, edgeType, count: crossMap[k] };
});

// ---- E. Inter-group import frequency ----
const interMap = {};
for (const e of importEdges) {
  const gs = idToGroup[e.source], gt = idToGroup[e.target];
  if (gs === undefined || gt === undefined || gs === gt) continue;
  const key = gs + "->" + gt;
  interMap[key] = (interMap[key] || 0) + 1;
}
const interGroupImports = Object.keys(interMap).map((k) => {
  const [from, to] = k.split("->");
  return { from, to, count: interMap[k] };
}).sort((a, b) => b.count - a.count);

// ---- F. Intra-group density ----
const intraGroupDensity = {};
const groupTotal = {};
const groupInternal = {};
for (const g in directoryGroups) { groupTotal[g] = 0; groupInternal[g] = 0; }
for (const e of importEdges) {
  const gs = idToGroup[e.source], gt = idToGroup[e.target];
  if (gs !== undefined) groupTotal[gs]++;
  if (gt !== undefined && gt !== gs) groupTotal[gt]++;
  if (gs !== undefined && gs === gt) { groupInternal[gs]++; }
}
for (const g in directoryGroups) {
  const total = groupTotal[g];
  const internal = groupInternal[g];
  intraGroupDensity[g] = { internalEdges: internal, totalEdges: total, density: total ? +(internal / total).toFixed(3) : 0 };
}

// ---- G. Pattern matching ----
const dirPatterns = [
  [/^(routes|api|controllers|endpoints|handlers|controller|routers|blueprints|serializers)$/, "api"],
  [/^(services|core|lib|domain|logic|signals|composables|mailers|jobs|channels)$/, "service"],
  [/^(models|db|data|persistence|repository|entities|migrations|entity|sql|database)$/, "data"],
  [/^(components|views|pages|ui|layouts|screens)$/, "ui"],
  [/^(middleware|plugins|interceptors|guards)$/, "middleware"],
  [/^(utils|helpers|common|shared|tools|templatetags|pkg)$/, "utility"],
  [/^(config|constants|env|settings|management|commands)$/, "config"],
  [/^(__tests__|test|tests|spec|specs)$/, "test"],
  [/^(types|interfaces|schemas|contracts|dtos|dto|request|response)$/, "types"],
  [/^hooks$/, "hooks"],
  [/^(store|state|reducers|actions|slices)$/, "state"],
  [/^(assets|static|public)$/, "assets"],
  [/^(cmd|bin|internal)$/, "entry"],
  [/^(docs|documentation|wiki)$/, "documentation"],
  [/^(deploy|deployment|infra|infrastructure|k8s|kubernetes|helm|charts|terraform|tf|docker)$/, "infrastructure"],
  [/^(\.github|\.gitlab|\.circleci)$/, "ci-cd"],
];
function matchDir(name) {
  for (const [re, label] of dirPatterns) if (re.test(name)) return label;
  return null;
}
function matchFile(n) {
  const fp = n.filePath || "";
  const base = n.name || fp.split("/").pop();
  if (/\.(test|spec)\.[^.]+$/.test(base) || /^test_.*\.py$/.test(base) || /_test\.go$/.test(base) || /Test\.java$/.test(base) || /_spec\.rb$/.test(base) || /Test\.php$/.test(base) || /Tests\.cs$/.test(base)) return "test";
  if (/\.d\.ts$/.test(base)) return "types";
  if (base === "manage.py") return "entry";
  if (base === "wsgi.py" || base === "asgi.py") return "config";
  if (/^(Dockerfile|docker-compose.*)$/.test(base)) return "infrastructure";
  if (/\.(tf|tfvars)$/.test(base)) return "infrastructure";
  if (/\.gitlab-ci\.yml$/.test(base) || base === "Jenkinsfile") return "ci-cd";
  if (fp.startsWith(".github/workflows/")) return "ci-cd";
  if (/\.sql$/.test(base)) return "data";
  if (/\.(graphql|gql|proto)$/.test(base)) return "types";
  if (/\.(md|rst)$/.test(base)) return "documentation";
  if (base === "Makefile") return "infrastructure";
  if (/^(Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle|composer\.json)$/.test(base)) return "config";
  return null;
}
const patternMatches = {};
for (const g in directoryGroups) {
  const m = matchDir(g);
  if (m) patternMatches[g] = m;
}

// ---- H. Deployment topology ----
let hasDockerfile = false, hasCompose = false, hasK8s = false, hasTerraform = false, hasCI = false;
const infraFiles = [];
for (const n of fileNodes) {
  const fp = n.filePath || "";
  const base = n.name || "";
  if (/^Dockerfile/.test(base)) { hasDockerfile = true; infraFiles.push(fp); }
  else if (/^docker-compose/.test(base)) { hasCompose = true; infraFiles.push(fp); }
  else if (/\.(ya?ml)$/.test(base) && /(k8s|kube|deployment|helm)/i.test(fp)) { hasK8s = true; infraFiles.push(fp); }
  else if (/\.(tf|tfvars)$/.test(base)) { hasTerraform = true; infraFiles.push(fp); }
  else if (fp.startsWith(".github/workflows/") || /\.gitlab-ci/.test(base) || base === "Jenkinsfile") { hasCI = true; infraFiles.push(fp); }
}
const deploymentTopology = { hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI, infraFiles };

// ---- I. Data pipeline ----
const dataPipeline = { schemaFiles: [], migrationFiles: [], dataModelFiles: [], apiHandlerFiles: [] };
for (const n of fileNodes) {
  const fp = n.filePath || "";
  if (/\.(sql)$/.test(fp) || /\.(graphql|gql|proto|prisma)$/.test(fp)) dataPipeline.schemaFiles.push(fp);
  if (/migrations?\//.test(fp)) dataPipeline.migrationFiles.push(fp);
  if (/\/(models|db)\//.test(fp) || /models?\.py$/.test(fp)) dataPipeline.dataModelFiles.push(fp);
  if (/\/(api|routes|routers|controllers)\//.test(fp)) dataPipeline.apiHandlerFiles.push(fp);
}

// ---- J. Doc coverage ----
const docFiles = fileNodes.filter((n) => /\.(md|rst)$/.test(n.filePath || ""));
const groupsWithDocsSet = new Set();
for (const d of docFiles) groupsWithDocsSet.add(idToGroup[d.id]);
const totalGroups = Object.keys(directoryGroups).length;
const undocumentedGroups = Object.keys(directoryGroups).filter((g) => !groupsWithDocsSet.has(g));
const docCoverage = {
  groupsWithDocs: groupsWithDocsSet.size,
  totalGroups,
  coverageRatio: totalGroups ? +(groupsWithDocsSet.size / totalGroups).toFixed(2) : 0,
  undocumentedGroups,
};

// ---- K. Dependency direction ----
const dependencyDirection = [];
const seenPair = new Set();
for (const e of interGroupImports) {
  const a = e.from, b = e.to;
  const key = [a, b].sort().join("|");
  if (seenPair.has(key)) continue;
  seenPair.add(key);
  const ab = interMap[a + "->" + b] || 0;
  const ba = interMap[b + "->" + a] || 0;
  if (ab >= ba && ab > 0) dependencyDirection.push({ dependent: a, dependsOn: b });
  else if (ba > 0) dependencyDirection.push({ dependent: b, dependsOn: a });
}

// ---- File stats ----
const filesPerGroup = {};
for (const g in directoryGroups) filesPerGroup[g] = directoryGroups[g].length;
const nodeTypeCounts = {};
for (const t in nodeTypeGroups) nodeTypeCounts[t] = nodeTypeGroups[t].length;

const result = {
  scriptCompleted: true,
  commonPrefix: prefix,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats: { totalFileNodes: fileNodes.length, filesPerGroup, nodeTypeCounts },
  fileFanIn: fanIn,
  fileFanOut: fanOut,
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
} catch (e) {
  console.error("Failed to write output: " + e.message);
  process.exit(1);
}
console.log("Analysis complete. Groups: " + Object.keys(directoryGroups).join(", "));
process.exit(0);
