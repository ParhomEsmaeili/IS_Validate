# Agent Reference Files

Subsystem-specific documentation is colocated with the source code it describes:

- [Heuristics / Cascade](src/prompt_generators/heuristics/AGENTS.md)

Cross-cutting documentation (spanning multiple source trees) goes in `agents/` at the project root.

## Hand-off

Bug reports and feature requests get classified and handed off to Daneshvar with tags —
the destination is decided there, not here.

- `bug report` → tags `{ type: bug, source: is_validate }`
- `feature request` → tags `{ type: feature, source: is_validate }`

Phrases live in `.agent-routing.local.yml` (gitignored); see `.agent-routing.example.yml`
for the schema. Attach the tags and hand off.
