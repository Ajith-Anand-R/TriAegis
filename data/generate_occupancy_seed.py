from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.occupancy import OccupancyEngine


def main() -> None:
    engine = OccupancyEngine()
    summary = engine.seed_from_file(reset=True)
    load = engine.current_load()["summary"]

    print("Occupancy seed completed")
    print(
        (
            f"Hospitals={summary['hospital_count']} "
            f"Wards={summary['ward_count']} "
            f"Beds={summary['bed_count']}"
        )
    )
    print(
        (
            f"NetworkLoad={load['network_load_percent']}% "
            f"CriticalWards={load['critical_ward_count']} "
            f"WarningWards={load['warning_ward_count']}"
        )
    )


if __name__ == "__main__":
    main()
