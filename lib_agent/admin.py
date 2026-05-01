# admin.py — small CLI for managing persisted state.
#
# Two stores get cleaned up here:
#   - LangGraph checkpoints (checkpoints.sqlite) — one row per super-step per thread
#   - Phoenix traces (Docker volume) — every span from every run
#
# Usage from lib_agent/:
#   python -m admin info
#   python -m admin list-threads
#   python -m admin purge-thread <thread_id>
#   python -m admin purge-all       --yes
#   python -m admin purge-traces    --yes

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DB_PATH = ROOT / "checkpoints.sqlite"
PHOENIX_COMPOSE = ROOT / "observability" / "docker-compose.phoenix.yml"


def _connect():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH))


def cmd_info(_args) -> int:
    print(f"checkpoint db: {DB_PATH}  ({'exists' if DB_PATH.exists() else 'absent'})")
    if DB_PATH.exists():
        size = DB_PATH.stat().st_size
        print(f"  size: {size:,} bytes")
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
        ).fetchone()
        print(f"  threads: {rows[0]}")

    print(f"\nphoenix compose: {PHOENIX_COMPOSE}")
    res = subprocess.run(
        ["docker", "ps", "--filter", "name=phoenix", "--format", "{{.Status}}"],
        capture_output=True, text=True,
    )
    print(f"  container: {res.stdout.strip() or 'not running'}")
    return 0


def cmd_list_threads(_args) -> int:
    conn = _connect()
    if conn is None:
        print("no checkpoint db (yet)")
        return 0
    rows = conn.execute(
        "SELECT thread_id, COUNT(*) AS n FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
    ).fetchall()
    if not rows:
        print("(no threads)")
        return 0
    print(f"{'thread_id':<32}  checkpoints")
    for tid, n in rows:
        print(f"{tid:<32}  {n}")
    return 0


def cmd_purge_thread(args) -> int:
    conn = _connect()
    if conn is None:
        print("no checkpoint db", file=sys.stderr)
        return 1
    n_ck = conn.execute(
        "DELETE FROM checkpoints WHERE thread_id = ?", (args.thread_id,)
    ).rowcount
    n_wr = conn.execute(
        "DELETE FROM writes WHERE thread_id = ?", (args.thread_id,)
    ).rowcount
    conn.commit()
    print(f"deleted {n_ck} checkpoint rows, {n_wr} write rows for thread_id={args.thread_id!r}")
    return 0


def cmd_purge_all(args) -> int:
    if not args.yes:
        print(f"refusing without --yes: would delete {DB_PATH}", file=sys.stderr)
        return 1
    if DB_PATH.exists():
        os.remove(DB_PATH)
        print(f"deleted {DB_PATH}")
    else:
        print("nothing to delete")
    return 0


def cmd_purge_traces(args) -> int:
    """Nuke the Phoenix volume. ALL traces from ALL projects are lost.
    For local dev this is fine — restart is fast and there's only one project."""
    if not args.yes:
        print("refusing without --yes: would `docker compose down -v` Phoenix", file=sys.stderr)
        return 1
    cmd_down = ["docker", "compose", "-f", str(PHOENIX_COMPOSE), "down", "-v"]
    cmd_up = ["docker", "compose", "-f", str(PHOENIX_COMPOSE), "up", "-d"]
    print("$ " + " ".join(cmd_down))
    subprocess.run(cmd_down, check=True)
    print("$ " + " ".join(cmd_up))
    subprocess.run(cmd_up, check=True)
    print("phoenix restarted with empty data store")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="admin", description="manage persisted state")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show db + phoenix status")
    sub.add_parser("list-threads", help="list thread_ids and checkpoint counts")

    sp = sub.add_parser("purge-thread", help="delete a single thread")
    sp.add_argument("thread_id")

    sp = sub.add_parser("purge-all", help="delete the entire checkpoint db")
    sp.add_argument("--yes", action="store_true", help="confirm destructive action")

    sp = sub.add_parser("purge-traces", help="nuke Phoenix data (volume reset)")
    sp.add_argument("--yes", action="store_true", help="confirm destructive action")

    args = p.parse_args()
    handlers = {
        "info": cmd_info,
        "list-threads": cmd_list_threads,
        "purge-thread": cmd_purge_thread,
        "purge-all": cmd_purge_all,
        "purge-traces": cmd_purge_traces,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
