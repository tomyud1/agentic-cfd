# Instructions for coding agents

Every coding agent starts here. The full instructions are in [`CLAUDE.md`](CLAUDE.md), and they
apply to every agent (Claude Code, Codex and others), not only Claude. Read it first.

- What changed recently: [`CHANGELOG.md`](CHANGELOG.md).
- To solve a CFD problem with the project's workflow, follow
  [`.claude/skills/agentic-cfd/SKILL.md`](.claude/skills/agentic-cfd/SKILL.md) and its
  `references/` folder. Agents other than Claude Code don't load it automatically, so open it by
  path.
- Machine setup (WSL2, OpenFOAM, the Python environment):
  [`.claude/skills/agentic-cfd/references/environment.md`](.claude/skills/agentic-cfd/references/environment.md)
  and [`README.md`](README.md), "Reproduce it".
