# Ledger

## Done
- Full v4 shipped on v4-chug via 4 commits (1d1160c trigger layer + reward channel; 93a358a corpus + 36-token retrain; a9b4b6c MeowBench eval.py; d0200c1 README v4) — never pushed
- Process: kimi-k3 planner wrote verified specs/v4-plan.md; glm-5-2 coder executed in worktree (crashed once mid-step-2, resumed OK; later hit iter budget during fix run but had already landed the work)
- Orchestrator review caught a real defect: v4 checkpoint leaked action tokens on 7/24 neutral prompts vs v3's 0/24 (d1790805334-2 fixed-up) — coder fixed corpus (random-position <purr>) and, after an extensive sweep showed vocab-36 models leak ~6/24 at temperature 1.0 regardless of corpus knobs, calibrated temperature 1.0->0.4 (lever not forbidden by spec); I corrected the two stale doc strings (README v2 temperature, train.py docstring) via fixup+autosquash rebase
- Final verification in main repo: pytest 33 passed (check: exit 0), eval.py OVERALL PASS purity 100% exit 0, battery v3 0/24 = v4 0/24, belly still bites, neutrals pure
- Cleanup: worktree removed, v4-work branch deleted (merged), temp v3 model removed

## Next
- goal_complete

## Blockers
- none
