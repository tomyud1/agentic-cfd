# Instructions for coding agents

This repository's agent instructions are in [`CLAUDE.md`](CLAUDE.md). They apply to every coding
agent (Claude Code, Codex and others). Read it first.

- What changed recently: [`CHANGELOG.md`](CHANGELOG.md).
- To solve a CFD problem with the project's workflow, follow
  [`.claude/skills/agentic-cfd/SKILL.md`](.claude/skills/agentic-cfd/SKILL.md) and its
  `references/` folder. Agents other than Claude Code don't load it automatically, so open it by
  path.
- Machine setup (WSL2, OpenFOAM, the Python environment):
  [`.claude/skills/agentic-cfd/references/environment.md`](.claude/skills/agentic-cfd/references/environment.md)
  and [`README.md`](README.md), "Reproduce it".
