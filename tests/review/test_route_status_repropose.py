"""helpers/route_status.py --todo=repropose: the drives Propose again works on (helpers/1_prepare.sh --again)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from nnslr_tools.review_db import ReviewStore, db_path, save_frame_review
from review_helpers import ROUTE, RUN_ID, make_report, page_body

REPO = Path(__file__).resolve().parents[2]


def repropose(root: Path) -> list[str]:
    env = {**os.environ, 'NNSLR_DATA_ROOT': str(root), 'COMMA_VIDEOS_DIR': str(root / 'videos')}
    done = subprocess.run([sys.executable, '-m', 'helpers.route_status', '--todo=repropose'], cwd=REPO, env=env,
                          capture_output=True, text=True, check=True)
    return done.stdout.split()


def test_a_drive_in_review_is_proposed_again_until_every_frame_is_confirmed(tmp_path):
    rows = make_report(tmp_path)
    assert repropose(tmp_path) == [ROUTE]
    store = ReviewStore(db_path(tmp_path))
    for count, row in enumerate(rows, start=1):
        save_frame_review(store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=row, body=page_body(row),
                          client='test')
        # Frames still to confirm: it stays on the list; all confirmed: nothing to propose again.
        assert repropose(tmp_path) == ([ROUTE] if count < len(rows) else [])
