Agent Implementation Prompt — Use to Execute a Ticket End‑to‑End

When to use: Run this in Codex CLI to implement a single ticket from start to finish on a feature branch, with logs, tests, and a JSON summary.

Inputs (fill these placeholders):
- {{NNN}} — ticket number (e.g., 011)
- {{slug}} — short hyphenated slug (e.g., prompts-docs-and-readme-updates)
- {{branch}} — feature branch name (e.g., feat/gpt5-011-prompts-docs)
- {{ticketFile}} — path to ticket file (e.g., ./codex/tickets/011-prompts-docs-and-readme-updates.md)
- {{logFile}} — path to log file (e.g., ./codex/logs/011.md)

Task Framing
- Read {{ticketFile}} and AGENTS.md to confirm scope and acceptance checks.
- Never change runtime behavior unless the ticket explicitly requires it.
- Prefer documentation, prompts, tests, and helper templates over code changes.
- Keep diffs minimal and scoped.

Branching
- Base: feat/gpt5-010-rollout-flag (stacked branches).
- Create: {{branch}}.
- Only stage and commit files touched for this ticket.

Operational Steps (no chain of thought; operational steps only)
1) Read {{ticketFile}} and AGENTS.md. In {{logFile}}, write a concise action plan (4–7 bullets).
2) Implement the ticket (docs/prompts/tests only if specified; no behavior changes).
3) Add Playwright tests for Node-only, fs-based checks if specified by the ticket.
4) Update {{logFile}} with: What I Did, Commands Run, Files Touched, Test Results, Next Steps.
5) Commit with the mandated commit message if provided in the ticket.

RUN & ITERATE UNTIL GREEN
- Detect package manager by lockfile:
  - If pnpm-lock.yaml exists: `pnpm install && pnpm test`
  - Else if yarn.lock exists: `yarn && yarn test`
  - Else: `npm install && npm run test`
- Run tests with `playwright test` via the package script.
- If tests fail, FIX and re-run. Iterate up to 5 times or until all tests pass.
- Keep network disabled in tests; only read files via `fs`.

Playwright Test Invocation
- The repo uses single-project Node-only config in `playwright.config.ts`.
- Add tests under `./tests/` with `.spec.ts` using `@playwright/test` APIs.
- Avoid SDK imports or any network calls; restrict to `fs` and string checks.

Logging Discipline
- Append succinct, operational entries to {{logFile}} only.
- Do not include chain-of-thought; list commands, actions, and results.

Acceptance Checks
- Re-read {{ticketFile}} acceptance section and ensure every check is met.
- Ensure no runtime/provider defaults are flipped unless explicitly allowed.

Final Output (print as last line only)
{
  "ticket": "{{NNN}}",
  "branch": "{{branch}}",
  "status": "<success|needs-attention>",
  "changed_files": ["<paths>"],
  "tests": {"runner":"playwright","docs":"<pass|fail>"},
  "notes": "<1-2 line summary or blocker>"
}

BEGIN.

