"""link-videos: link segments already in the shared videos folder, no network."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nnslr_tools import cli
from nnslr_tools.sync import DEFAULT_VIDEOS_DIR

ROUTE = "000001a3--c20ba54385"
OTHER = "0000011e--0c8825628a"


def make_segment(videos: Path, name: str, files: tuple[str, ...]) -> Path:
    directory = videos / name
    directory.mkdir(parents=True)
    for file_name in files:
        (directory / file_name).write_bytes(f"{name}:{file_name}".encode())
    return directory


@pytest.fixture
def shared(tmp_path, monkeypatch):
    # Same content as the folder openpilot.distill filled on the training LXC:
    # only fcamera.hevc, ecamera.hevc and rlog.zst per segment.
    videos = tmp_path / "comma videos"
    root = tmp_path / "data root"
    complete = ("fcamera.hevc", "ecamera.hevc", "rlog.zst")
    make_segment(videos, f"{ROUTE}--0", complete)
    make_segment(videos, f"{ROUTE}--2", complete)
    make_segment(videos, f"{ROUTE}--10", complete)
    make_segment(videos, f"{OTHER}--1", complete)
    make_segment(videos, f"{OTHER}--3", ("ecamera.hevc", "rlog.zst"))
    make_segment(videos, f"{OTHER}--4", ("fcamera.hevc",))
    (videos / "boot").mkdir()
    (videos / ".DS_Store").write_bytes(b"")
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(videos))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(root))
    return videos, root


def run(capfd, *args):
    try:
        code = cli.main(["link-videos", *args])
    except SystemExit as exc:
        code = exc.code
    captured = capfd.readouterr()
    return code, captured.out, captured.err


def snapshot(folder: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            result[str(path.relative_to(folder))] = path.read_bytes()
    return result


def test_all_routes_are_linked_and_missing_files_reported(shared, capfd):
    videos, root = shared
    before = snapshot(videos)
    code, out, err = run(capfd)
    assert code == 1, err  # missing files are reported, not a crash
    report = json.loads(out)
    assert report["ok"] is False
    assert report["videos_dir"] == str(videos)
    assert report["segments"] == [
        f"{OTHER}--1", f"{OTHER}--3", f"{OTHER}--4", f"{ROUTE}--0", f"{ROUTE}--2", f"{ROUTE}--10",
    ]
    assert report["missing_files"] == [
        {"route": OTHER, "segment": 3, "name": "fcamera.hevc"},
        {"route": OTHER, "segment": 4, "name": "rlog"},
    ]
    assert report["ignored"] == ["boot"]
    assert report["counts"] == {"created": 15}
    for index in (0, 2, 10):
        for name in ("fcamera.hevc", "ecamera.hevc", "rlog.zst"):
            link = root / "raw/routes" / ROUTE / str(index) / name
            assert link.is_symlink()
            assert os.readlink(link) == str(videos / f"{ROUTE}--{index}" / name)
            assert link.read_bytes() == f"{ROUTE}--{index}:{name}".encode()
    # The files that exist in incomplete segments are still linked.
    assert (root / "raw/routes" / OTHER / "3/ecamera.hevc").is_symlink()
    assert (root / "raw/routes" / OTHER / "4/fcamera.hevc").is_symlink()
    assert not (root / "raw/routes" / OTHER / "3/fcamera.hevc").exists()
    assert snapshot(videos) == before  # the shared folder is only read


def test_one_route_only(shared, capfd):
    _, root = shared
    code, out, err = run(capfd, "--route", ROUTE)
    assert code == 0, err
    report = json.loads(out)
    assert report["ok"] is True
    assert report["route"] == ROUTE
    assert report["segments"] == [f"{ROUTE}--0", f"{ROUTE}--2", f"{ROUTE}--10"]
    assert sorted(p.name for p in (root / "raw/routes").iterdir()) == [ROUTE]


def test_dry_run_writes_nothing(shared, capfd):
    _, root = shared
    code, out, err = run(capfd, "--route", ROUTE, "--dry-run")
    assert code == 0, err
    report = json.loads(out)
    assert report["dry_run"] is True
    assert report["counts"] == {"planned": 9}
    assert not root.exists()


def test_rerun_keeps_existing_links(shared, capfd):
    assert run(capfd, "--route", ROUTE)[0] == 0
    code, out, err = run(capfd, "--route", ROUTE)
    assert code == 0, err
    assert json.loads(out)["counts"] == {"existing": 9}
    code, out, err = run(capfd, "--route", ROUTE, "--dry-run")
    assert code == 0, err
    assert json.loads(out)["counts"] == {"existing": 9}


def test_explicit_folders_override_the_environment(shared, tmp_path, capfd, monkeypatch):
    videos, _ = shared
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(tmp_path / "wrong"))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(tmp_path / "wrong root"))
    explicit_root = tmp_path / "explicit root"
    code, out, err = run(capfd, "--route", ROUTE, "--videos-dir", str(videos), "--data-root", str(explicit_root))
    assert code == 0, err
    assert (explicit_root / "raw/routes" / ROUTE / "0/fcamera.hevc").is_symlink()
    assert not (tmp_path / "wrong root").exists()
    # The readers would reject these links until COMMA_VIDEOS_DIR points to videos.
    warning = json.loads(out)["warning"]
    assert "readers only accept links into COMMA_VIDEOS_DIR" in warning
    assert str(tmp_path / "wrong") in warning
    assert f"export COMMA_VIDEOS_DIR='{videos}'" in warning
    assert "warning: link-videos:" in err


def test_no_warning_when_videos_dir_is_the_readers_folder(shared, capfd):
    videos, _ = shared
    code, out, err = run(capfd, "--route", ROUTE, "--videos-dir", str(videos))
    assert code == 0, err
    assert json.loads(out)["warning"] is None
    assert "warning" not in err


def test_old_regular_copies_in_raw_routes(shared, capfd):
    videos, root = shared
    segment_dir = root / "raw/routes" / ROUTE / "0"
    segment_dir.mkdir(parents=True)
    # Identical old copy: replaced by a link. Different size: refused, kept.
    (segment_dir / "fcamera.hevc").write_bytes(f"{ROUTE}--0:fcamera.hevc".encode())
    (segment_dir / "rlog.zst").write_bytes(b"a different, longer log file")
    code, out, _ = run(capfd, "--route", ROUTE)
    assert code == 1
    report = json.loads(out)
    rows = {(row["segment"], row["name"]): row for row in report["files"]}
    assert rows[(0, "fcamera.hevc")]["status"] == "replaced_duplicate"
    assert rows[(0, "rlog.zst")]["status"] == "failed"
    assert "size differs" in rows[(0, "rlog.zst")]["error"]
    assert (segment_dir / "fcamera.hevc").is_symlink()
    assert not (segment_dir / "rlog.zst").is_symlink()
    assert (segment_dir / "rlog.zst").read_bytes() == b"a different, longer log file"
    assert (videos / f"{ROUTE}--0/rlog.zst").read_bytes() == f"{ROUTE}--0:rlog.zst".encode()
    # One conflict does not stop the other segments.
    assert (root / "raw/routes" / ROUTE / "10/rlog.zst").is_symlink()
    assert report["counts"] == {"created": 7, "failed": 1, "replaced_duplicate": 1}


def make_conflicts(videos: Path, root: Path, tmp_path: Path) -> Path:
    segment_dir = root / "raw/routes" / ROUTE / "0"
    segment_dir.mkdir(parents=True)
    # Same size as the shared copy, different size, a live link elsewhere, and a
    # dangling link left after the shared folder was moved.
    (segment_dir / "fcamera.hevc").write_bytes(f"{ROUTE}--0:fcamera.hevc".encode())
    (segment_dir / "rlog.zst").write_bytes(b"a different, longer log file")
    other = tmp_path / "other.hevc"
    other.write_bytes(b"other")
    (segment_dir / "ecamera.hevc").symlink_to(other)
    second = root / "raw/routes" / ROUTE / "2"
    second.mkdir(parents=True)
    (second / "fcamera.hevc").symlink_to(tmp_path / "old place" / f"{ROUTE}--2" / "fcamera.hevc")
    return segment_dir


def test_dry_run_predicts_conflicts_and_writes_nothing(shared, tmp_path, capfd):
    videos, root = shared
    segment_dir = make_conflicts(videos, root, tmp_path)
    before = sorted((str(p), p.is_symlink()) for p in root.rglob("*"))
    code, out, err = run(capfd, "--route", ROUTE, "--dry-run")
    assert code == 1, err  # no misleading all-clear
    report = json.loads(out)
    assert report["ok"] is False
    rows = {(row["segment"], row["name"]): row for row in report["files"]}
    assert rows[(0, "fcamera.hevc")]["status"] == "would_replace_duplicate"
    assert rows[(0, "rlog.zst")]["status"] == "conflict"
    assert "size differs" in rows[(0, "rlog.zst")]["error"]
    assert rows[(0, "ecamera.hevc")]["status"] == "conflict"
    assert "points to" in rows[(0, "ecamera.hevc")]["error"]
    assert rows[(2, "fcamera.hevc")]["status"] == "would_relink"
    assert report["counts"] == {"conflict": 2, "planned": 5, "would_relink": 1, "would_replace_duplicate": 1}
    assert sorted((str(p), p.is_symlink()) for p in root.rglob("*")) == before
    assert not (segment_dir / "fcamera.hevc").is_symlink()


def test_real_run_matches_the_dry_run(shared, tmp_path, capfd):
    videos, root = shared
    segment_dir = make_conflicts(videos, root, tmp_path)
    code, out, _ = run(capfd, "--route", ROUTE)
    assert code == 1
    report = json.loads(out)
    rows = {(row["segment"], row["name"]): row for row in report["files"]}
    assert rows[(0, "fcamera.hevc")]["status"] == "replaced_duplicate"
    assert "error" not in rows[(0, "fcamera.hevc")]
    assert rows[(0, "rlog.zst")]["status"] == "failed"
    assert rows[(0, "ecamera.hevc")]["status"] == "failed"
    assert rows[(2, "fcamera.hevc")]["status"] == "relinked"
    assert os.readlink(root / "raw/routes" / ROUTE / "2/fcamera.hevc") == str(videos / f"{ROUTE}--2" / "fcamera.hevc")
    assert os.readlink(segment_dir / "ecamera.hevc") == str(tmp_path / "other.hevc")
    assert report["counts"] == {"created": 5, "failed": 2, "relinked": 1, "replaced_duplicate": 1}


def test_moved_shared_folder_is_relinked(shared, tmp_path, capfd, monkeypatch):
    videos, root = shared
    assert run(capfd, "--route", ROUTE)[0] == 0
    moved = tmp_path / "new disk" / "comma-videos"
    moved.parent.mkdir()
    videos.rename(moved)
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(moved))
    code, out, err = run(capfd, "--route", ROUTE)
    assert code == 0, err
    assert json.loads(out)["counts"] == {"relinked": 9}
    link = root / "raw/routes" / ROUTE / "0/fcamera.hevc"
    assert os.readlink(link) == str(moved / f"{ROUTE}--0" / "fcamera.hevc")


def test_symlink_loop_is_reported_not_a_crash(shared, capfd):
    _, root = shared
    segment_dir = root / "raw/routes" / ROUTE / "0"
    segment_dir.mkdir(parents=True)
    (segment_dir / "fcamera.hevc").symlink_to(segment_dir / "fcamera.hevc")
    for args in (("--route", ROUTE, "--dry-run"), ("--route", ROUTE)):
        code, out, err = run(capfd, *args)
        assert code == 1
        assert "Traceback" not in err
        rows = {(row["segment"], row["name"]): row for row in json.loads(out)["files"]}
        assert "symlink loop" in rows[(0, "fcamera.hevc")]["error"]
    assert rows[(0, "fcamera.hevc")]["status"] == "failed"
    assert (root / "raw/routes" / ROUTE / "2/fcamera.hevc").is_symlink()


def test_symlinked_route_folder_in_data_root_is_refused(shared, tmp_path, capfd):
    _, root = shared
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "raw/routes").mkdir(parents=True)
    (root / "raw/routes" / ROUTE).symlink_to(outside, target_is_directory=True)
    code, _, err = run(capfd)
    assert code == 2
    assert "destination_symlink" in err
    assert list(outside.iterdir()) == []
    assert not (root / "raw/routes" / OTHER).exists()  # nothing linked before the refusal


def test_unknown_route_missing_folder_and_bad_route_are_errors(shared, tmp_path, capfd):
    code, _, err = run(capfd, "--route", "000001a4--abcdef1234")
    assert code == 2
    assert "route_not_found" in err
    code, _, err = run(capfd, "--videos-dir", str(tmp_path / "nowhere"))
    assert code == 2
    assert "videos_dir_not_found" in err
    code, _, err = run(capfd, "--route", "../escape")
    assert code == 2


def test_env_reports_videos_dir(capfd, monkeypatch, tmp_path):
    monkeypatch.delenv("COMMA_VIDEOS_DIR", raising=False)
    assert cli.main(["env"]) == 0
    report = json.loads(capfd.readouterr().out)
    assert report["comma_videos_dir"]["configured"] is False
    assert report["comma_videos_dir"]["path"] == str(Path(DEFAULT_VIDEOS_DIR).resolve())
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(tmp_path))
    assert cli.main(["env"]) == 0
    report = json.loads(capfd.readouterr().out)
    assert report["comma_videos_dir"] == {
        "configured": True, "path": str(tmp_path), "exists": True,
        "note": report["comma_videos_dir"]["note"],
    }
