import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.measurement_error.phase_distance_vs_order_common import DECIMAL_METRIC, main


if __name__ == "__main__":
    main(DECIMAL_METRIC)
