from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.database import init_db


def main() -> None:
    path = init_db()
    print(f"SQLite database initialized at {path}")


if __name__ == "__main__":
    main()
