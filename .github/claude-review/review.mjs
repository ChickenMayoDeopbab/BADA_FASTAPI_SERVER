import { appendFileSync, readFileSync } from "node:fs";
import { createSdkMcpServer, query, tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const { REPO, PR, HEAD_SHA, GH_TOKEN, RESULT_FILE } = process.env;
const DRY_RUN = process.env.DRY_RUN === "1";
const MODEL = process.env.REVIEW_MODEL || "claude-sonnet-5";
const MAX_INLINE = 6;

for (const [k, v] of Object.entries({ REPO, PR, HEAD_SHA })) {
  if (!v) throw new Error(`${k} 환경변수가 없다`);
}
if (!DRY_RUN && !GH_TOKEN) throw new Error("GH_TOKEN 환경변수가 없다");

let inlineCount = 0;
let summaryUrl = null;

async function github(method, path, body) {
  const res = await fetch(`https://api.github.com${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${GH_TOKEN}`,
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(`GitHub ${res.status}: ${data.message ?? ""} ${JSON.stringify(data.errors ?? "")}`);
  }
  return data;
}

const ok = (text) => ({ content: [{ type: "text", text }] });
const fail = (text) => ({ content: [{ type: "text", text }], isError: true });

const inlineComment = tool(
  "create_inline_comment",
  `PR 의 특정 라인에 인라인 코멘트를 단다. 최대 ${MAX_INLINE}개. line 은 PR 헤드 기준 새 파일의 줄 번호이고 diff 안에 있어야 한다.`,
  {
    path: z.string().describe("저장소 루트 기준 파일 경로"),
    line: z.number().int().positive().describe("코멘트할 줄(여러 줄이면 마지막 줄)"),
    start_line: z.number().int().positive().optional().describe("여러 줄 코멘트의 첫 줄"),
    body: z.string().min(1).describe("코멘트 본문(마크다운)"),
  },
  async ({ path, line, start_line, body }) => {
    if (inlineCount >= MAX_INLINE) {
      return fail(`인라인 코멘트는 최대 ${MAX_INLINE}개다. 나머지는 총평에 묶는다.`);
    }
    inlineCount += 1;
    const payload = { body, commit_id: HEAD_SHA, path, line, side: "RIGHT" };
    if (start_line && start_line < line) Object.assign(payload, { start_line, start_side: "RIGHT" });
    if (DRY_RUN) {
      console.log(`[DRY_RUN] inline ${JSON.stringify(payload)}`);
      return ok(`https://github.com/${REPO}/pull/${PR}#dry-run-inline-${inlineCount}`);
    }
    try {
      const data = await github("POST", `/repos/${REPO}/pulls/${PR}/comments`, payload);
      return ok(data.html_url);
    } catch (e) {
      inlineCount -= 1;
      return fail(`게시 실패: ${e.message}. diff 밖의 줄이면 총평에 넣는다.`);
    }
  },
);

const summaryComment = tool(
  "post_summary_comment",
  "PR 에 top-level 총평 코멘트를 게시한다. 리뷰당 한 번만 부른다. 성공하면 코멘트 URL 을 돌려준다.",
  { body: z.string().min(1).describe("총평 본문(마크다운)") },
  async ({ body }) => {
    if (summaryUrl) return fail(`총평은 이미 게시됐다: ${summaryUrl}`);
    if (DRY_RUN) {
      console.log(`[DRY_RUN] summary\n${body}`);
      summaryUrl = `https://github.com/${REPO}/pull/${PR}#dry-run-summary`;
      return ok(summaryUrl);
    }
    try {
      const data = await github("POST", `/repos/${REPO}/issues/${PR}/comments`, { body });
      summaryUrl = data.html_url;
      return ok(summaryUrl);
    } catch (e) {
      return fail(`게시 실패: ${e.message}`);
    }
  },
);

const prompt = readFileSync(new URL("./prompt.md", import.meta.url), "utf8")
  .replaceAll("{{REPO}}", REPO)
  .replaceAll("{{PR}}", PR)
  .replaceAll("{{MAX_INLINE}}", String(MAX_INLINE));

const run = query({
  prompt,
  options: {
    model: MODEL,
    effort: "low",
    cwd: process.env.GITHUB_WORKSPACE || process.cwd(),
    systemPrompt: { type: "preset", preset: "claude_code" },
    settingSources: [],
    strictMcpConfig: true,
    mcpServers: {
      github: createSdkMcpServer({ name: "github", version: "1.0.0", tools: [inlineComment, summaryComment] }),
    },
    permissionMode: "dontAsk",
    allowedTools: [
      "Read",
      "Glob",
      "Grep",
      "Bash(gh pr diff:*)",
      "Bash(gh pr view:*)",
      "Bash(git diff:*)",
      "Bash(git log:*)",
      "Bash(git show:*)",
      "mcp__github__create_inline_comment",
      "mcp__github__post_summary_comment",
    ],
    disallowedTools: ["Skill", "Task", "Agent", "Workflow", "ReportFindings", "Write", "Edit", "WebFetch", "WebSearch"],
    maxTurns: 40,
  },
});

let result = null;
for await (const message of run) {
  if (RESULT_FILE) appendFileSync(RESULT_FILE, `${JSON.stringify(message)}\n`);
  if (message.type === "system" && message.subtype === "init") {
    console.log(`init model=${message.model} apiKeySource=${message.apiKeySource}`);
  } else if (message.type === "assistant") {
    for (const block of message.message.content ?? []) {
      if (block.type === "tool_use") console.log(`tool ${block.name} ${JSON.stringify(block.input).slice(0, 200)}`);
    }
  } else if (message.type === "result") {
    result = message;
  }
}

console.log(
  JSON.stringify({
    subtype: result?.subtype,
    is_error: result?.is_error,
    num_turns: result?.num_turns,
    total_cost_usd: result?.total_cost_usd,
    permission_denials: result?.permission_denials?.length,
    inline_comments: inlineCount,
    summary_url: summaryUrl,
    result: String(result?.result ?? "").slice(0, 500),
  }),
);

if (!result || result.is_error) process.exit(1);
if (!summaryUrl) {
  console.error("총평이 게시되지 않았다");
  process.exit(1);
}
