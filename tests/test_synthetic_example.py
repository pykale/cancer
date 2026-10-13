import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic" / "synthetic_survival.py"


def test_synthetic_survival_example_runs_and_learns():
    result = subprocess.run([sys.executable, str(EXAMPLE)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    score_line = next(line for line in result.stdout.splitlines() if line.startswith("harrell_c"))
    harrell_c = float(score_line.split()[-1])
    assert harrell_c > 0.7, f"the synthetic example stopped learning (Harrell's C {harrell_c:.3f}); keep it above 0.7"
