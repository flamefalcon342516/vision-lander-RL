"""Step 15: distil the residual PPO policy (pose-based) into a pixel CNN with DAgger."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lander.config import load_all  # noqa: E402
from lander.eval.evaluate import eval_grid, load_policy, summarize, write_csv  # noqa: E402
from lander.stretch.pixels import dagger, evaluate_student  # noqa: E402

cfg = load_all()
teacher = load_policy(cfg, Path("results/runs/main_s0"))
out = Path("results/stretch")
student = dagger(cfg, teacher, out, iters=5, steps_per_iter=600, n_env=64, epochs=4)
rows = summarize(evaluate_student(cfg, student, eval_grid(cfg), episodes=200, seed=4242))
write_csv(rows, out / "student_fast_dr.csv")
print(json.dumps({r["path"] + "@" + str(r["speed"]): round(r["success"], 3) for r in rows}, indent=1))
