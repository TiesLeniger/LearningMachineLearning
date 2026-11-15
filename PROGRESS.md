# Progress

## Current
- Phase: 1 (ML fundamentals)
- Next step: 1.1 Learning framework (read B&B ch. 1-2, write `notes/phase1-learning-framework.md`)

## Log
- 2026-11-15: Project scoped. CLAUDE.md, README, topic map created.
- 2026-11-15: 0.1 done. uv-managed env, Python 3.12.16, torch 2.15.0+cpu, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, pytest 9.2.0. Smoke tests pass (6/6). See `phase0_setup/01_environment/`.
- 2026-11-15: 0.2 done (PR #1). Repo public, topics set. .gitattributes (LF), Conventional Commits + branch/PR/squash workflow in CLAUDE.md, GitHub Actions pytest CI. MIT LICENSE added.
- 2026-11-15: 0.3 done. Workflow set by CLAUDE.md, PROGRESS.md and the PR review loop; first real review at 1.2.
- 2026-11-15: 0.4 done (PR #3). OpenMDAO 3.45.2, Gymnasium 1.4.0, XFOIL 6.99, AVL 3.40 in `tools/` (gitignored). 12 smoke tests pass locally; CI skips XFOIL/AVL. Phase 0 complete.
- 2026-11-15: AVL CLa point closed: AVL (4.57 /rad, mesh-converged) is a higher-resolution lifting-surface result than lifting-line/Helmbold estimates (4.91-5.03); the difference is expected.
- 2026-11-15: 1.0 done. Backbone: Bishop & Bishop (R&W for GPs). Chapter mapping in topic map. Notes template, `notes/img/` for scanned derivations, VS Code LaTeX snippets (Tab to expand in Markdown).

## Open questions
- Verify B&B section numbers in the topic map against the book (mapping written from memory).
- Does current PyTorch (CUDA build) still support the GTX 1060 (Pascal)? Check before using the tower.
- JSBSim on native Windows or WSL2? Check at 4.5.

## Notes
- Test file names must be unique repo-wide (pytest default import mode).
- Autograd error was exactly 0 because the test compares identical float ops; a real gradient check (finite differences) comes at 1.7.
