Release Manager Prompt — Use for Stacked Branch Flow and PR Hygiene

When to use: Coordinate stacked PRs (001→…→012), keep branches rebased, ensure clean PR descriptions and checklists, and manage safe rollouts.

Branching & Rebasing Etiquette
- Branch naming: `feat/gpt5-{{NNN}}-{{slug}}` off previous ticket branch.
- Keep each branch focused; avoid cross-ticket edits.
- Rebase frequently onto the immediate predecessor branch, not `main`.
- If a predecessor changes, retarget dependent PRs accordingly.
- Never squash away meaningful history between stacked tickets.

PR Descriptions & Checklists
- Context: brief summary and scope of the ticket.
- Changes: list of files and high-level rationale.
- Tests: how to run (`npm|yarn|pnpm test`) and what is covered.
- Acceptance Checks: explicit list with PASS/FAIL.
- Risk/rollback: flags or kill switches if applicable.
- Link: ticket path (e.g., `./codex/tickets/{{NNN}}-{{slug}}.md`) and log (`./codex/logs/{{NNN}}.md`).

Rollout & Flags
- Respect rollout flags: killswitch, staging default, percent canary.
- Confirm no default flips are introduced unintentionally.
- Document operational playbooks in docs (flip/rollback steps, no code changes).

Operational Flow
1) Verify branch base matches the previous ticket branch.
2) Ensure only scoped files changed; ask implementers to split otherwise.
3) Confirm Playwright tests pass locally and in CI.
4) Merge stacked PRs in order; rebase dependents as each lands.
5) Trigger staging verification gates before canary or production.

BEGIN.

