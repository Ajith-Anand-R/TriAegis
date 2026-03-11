from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.healthcheck import run_healthcheck


def main() -> None:
    status = run_healthcheck(ROOT)

    print("=== TRIAGE SYSTEM HEALTHCHECK ===")
    print(json.dumps(status.to_dict(), indent=2))

    if not status.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
