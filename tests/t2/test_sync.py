# [nnslr-sync] - START
"""Exercise the CLI and real rsync protocol against local files, never a device."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from nnslr_tools import cli
from nnslr_tools.sync import DEFAULT_DATA_ROOT, DEFAULT_VIDEOS_DIR

ROUTE = "000001a3--c20ba54385"


@pytest.fixture
def source(tmp_path, monkeypatch):
    if shutil.which("rsync") is None:
        pytest.skip("local transfer integration tests require rsync")
    remote = tmp_path / "remote"
    remote.mkdir()
    commands = tmp_path / "ssh-calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Only SSH transport is replaced. The remote shell, file inventory and
    # both ends of rsync run for real on the test fixture's filesystem.
    ssh = bin_dir / "ssh"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "while args and args[0].startswith('-'):\n"
        "    option = args.pop(0)\n"
        "    if option in ('-o', '-l', '-p', '-i'): args.pop(0)\n"
        "args.pop(0)  # host\n"
        "command = ' '.join(args)\n"
        "with open(os.environ['TEST_SSH_CALLS'], 'a') as f: f.write(command + '\\n')\n"
        "if os.environ.get('TEST_SSH_FAIL'): sys.exit(255)\n"
        "if os.environ.get('TEST_TRANSFER_FAIL') and '--server' in command: sys.exit(12)\n"
        "command = command.replace('/data/media/0/realdata', os.environ['TEST_REMOTE'])\n"
        "os.execv('/bin/sh', ['sh', '-c', command])\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_REMOTE", str(remote))
    monkeypatch.setenv("TEST_SSH_CALLS", str(commands))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(tmp_path / "data with spaces"))
    # Never the real shared folder next to the clone.
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(tmp_path / "comma videos"))
    monkeypatch.delenv("NNSLR_COMMA_HOST", raising=False)
    return remote, Path(os.environ["NNSLR_DATA_ROOT"]), commands, Path(os.environ["COMMA_VIDEOS_DIR"])


def segment(remote: Path, index: int, *, route: str = ROUTE) -> Path:
    directory = remote / f"{route}--{index}"
    directory.mkdir()
    for name in ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc", "qcamera.ts", "rlog.zst", "qlog.zst"):
        (directory / name).write_bytes(f"{index}:{name}".encode())
    return directory


def shared_file(videos: Path, index: int, name: str) -> Path:
    return videos / f"{ROUTE}--{index}" / name


def raw_file(root: Path, index: int, name: str) -> Path:
    return root / "raw/routes" / ROUTE / str(index) / name


def assert_linked(root: Path, videos: Path, index: int, name: str) -> None:
    link = raw_file(root, index, name)
    target = shared_file(videos, index, name)
    assert link.is_symlink()
    assert os.readlink(link) == str(target)  # absolute link into the shared folder
    assert target.is_file() and not target.is_symlink()


def run(capfd, *args):
    try:
        code = cli.main(["sync-routes", "--route", ROUTE, *args])
    except SystemExit as exc:
        code = exc.code
    captured = capfd.readouterr()
    return code, captured.out, captured.err


def test_defaults_copy_into_shared_folder_and_link_into_raw_routes(source, capfd):
    remote, root, _, videos = source
    first = segment(remote, 0)
    segment(remote, 2)
    segment(remote, 1, route="000001a4--abcdef1234")
    code, out, err = run(capfd)
    assert code == 0, err
    report = json.loads(out)
    assert report["host"] == "comma"
    assert report["segments"] == [0, 2]
    assert report["ok"] is True
    assert report["videos_dir"] == str(videos)
    dest = root / "raw" / "routes" / ROUTE
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == [
        "0/fcamera.hevc", "0/rlog.zst", "2/fcamera.hevc", "2/rlog.zst",
    ]
    assert sorted(p.name for p in videos.iterdir()) == [f"{ROUTE}--0", f"{ROUTE}--2"]
    for index in (0, 2):
        for name in ("fcamera.hevc", "rlog.zst"):
            assert_linked(root, videos, index, name)
    assert (dest / "0/fcamera.hevc").read_bytes() == (first / "fcamera.hevc").read_bytes()
    row = report["files"][0]
    assert row["shared_path"] == str(shared_file(videos, 0, "fcamera.hevc"))
    assert row["destination"] == str(raw_file(root, 0, "fcamera.hevc"))
    assert row["status"] == "copied" and row["link_status"] == "created"
    assert row["sha256"] == hashlib.sha256(b"0:fcamera.hevc").hexdigest()
    assert (first / "dcamera.hevc").is_file()  # source is untouched


def test_dry_run_uses_fallback_defaults_without_creating_destination(source, capfd, monkeypatch):
    remote, _, _, _ = source
    segment(remote, 0)
    monkeypatch.delenv("NNSLR_DATA_ROOT")
    monkeypatch.delenv("COMMA_VIDEOS_DIR")
    code, out, err = run(capfd, "--dry-run")
    assert code == 0, err
    report = json.loads(out)
    assert report["data_root"] == str(Path(DEFAULT_DATA_ROOT).resolve())
    # <repo>/../comma-videos, the folder next to the clone.
    assert DEFAULT_VIDEOS_DIR == str(Path(DEFAULT_DATA_ROOT).parent.parent / "comma-videos")
    assert report["videos_dir"] == str(Path(DEFAULT_VIDEOS_DIR).resolve())
    assert report["dry_run"] is True
    assert all(row["status"] == "planned" and row["sha256"] is None for row in report["files"])
    assert report["files"][0]["shared_path"] == str(Path(DEFAULT_VIDEOS_DIR).resolve() / f"{ROUTE}--0" / "fcamera.hevc")


def test_overrides_select_segments_camera_host_and_destination(source, tmp_path, capfd):
    remote, default_root, _, default_videos = source
    for index in range(4):
        segment(remote, index)
    explicit = tmp_path / "explicit"
    explicit_videos = tmp_path / "explicit videos"
    code, out, err = run(capfd, "--host", "comma@192.0.2.10", "--segments", "1-2,2",
                         "--camera", "both", "--data-root", str(explicit), "--videos-dir", str(explicit_videos))
    assert code == 0, err
    report = json.loads(out)
    assert report["host"] == "comma@192.0.2.10"
    assert report["segments"] == [1, 2]
    assert not default_root.exists()
    assert not default_videos.exists()
    assert sorted(p.name for p in (explicit / "raw/routes" / ROUTE / "1").iterdir()) == [
        "ecamera.hevc", "fcamera.hevc", "rlog.zst",
    ]
    assert sorted(p.name for p in (explicit_videos / f"{ROUTE}--1").iterdir()) == [
        "ecamera.hevc", "fcamera.hevc", "rlog.zst",
    ]
    assert_linked(explicit, explicit_videos, 2, "ecamera.hevc")


def test_dry_run_does_not_create_files(source, capfd):
    remote, root, _, videos = source
    segment(remote, 0)
    code, out, err = run(capfd, "--dry-run", "--camera", "wide")
    assert code == 0, err
    assert not root.exists()
    assert not videos.exists()
    assert [row["name"] for row in json.loads(out)["files"]] == ["ecamera.hevc", "rlog.zst"]


def test_missing_video_full_log_and_requested_segment_are_reported(source, capfd):
    remote, _, _, _ = source
    directory = segment(remote, 0)
    (directory / "rlog.zst").unlink()
    (directory / "fcamera.hevc").unlink()
    code, out, _ = run(capfd, "--segments", "0-1", "--dry-run")
    assert code == 1
    report = json.loads(out)
    assert report["ok"] is False
    assert report["missing_segments"] == [1]
    assert {(row["segment"], row["name"]) for row in report["missing_files"]} == {(0, "fcamera.hevc"), (0, "rlog")}
    assert not report["files"]  # qlog must not silently substitute for rlog


@pytest.mark.parametrize("name", ["rlog", "rlog.bz2"])
def test_full_log_variants(source, capfd, name):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    (directory / "rlog.zst").rename(directory / name)
    code, _, err = run(capfd)
    assert code == 0, err
    assert (root / "raw/routes" / ROUTE / "0" / name).read_bytes() == b"0:rlog.zst"
    assert_linked(root, videos, 0, name)


def test_second_run_downloads_nothing_and_preserves_unrelated_files(source, capfd):
    remote, root, _, videos = source
    segment(remote, 0)
    assert run(capfd)[0] == 0
    dest = root / "raw/routes" / ROUTE / "0"
    (dest / "notes.txt").write_text("keep")
    before = shared_file(videos, 0, "fcamera.hevc").stat()
    code, out, err = run(capfd)
    assert code == 0, err
    rows = json.loads(out)["files"]
    assert all(row["status"] == "existing" and row["link_status"] == "existing" for row in rows)
    after = shared_file(videos, 0, "fcamera.hevc").stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert (dest / "notes.txt").read_text() == "keep"


def test_complete_shared_copy_from_another_project_is_linked_without_download(source, capfd):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    # e.g. openpilot.distill already downloaded this segment into the shared folder.
    shared = videos / f"{ROUTE}--0"
    shared.mkdir(parents=True)
    shutil.copy2(directory / "fcamera.hevc", shared / "fcamera.hevc")
    shutil.copy2(directory / "rlog.zst", shared / "rlog.zst")
    before = (shared / "fcamera.hevc").stat().st_ino
    code, out, err = run(capfd)
    assert code == 0, err
    rows = json.loads(out)["files"]
    assert [(row["status"], row["link_status"]) for row in rows] == [("existing", "created"), ("existing", "created")]
    assert (shared / "fcamera.hevc").stat().st_ino == before
    assert_linked(root, videos, 0, "fcamera.hevc")
    assert_linked(root, videos, 0, "rlog.zst")


def test_old_regular_files_in_raw_routes_move_into_shared_folder(source, capfd):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    shutil.copy2(directory / "fcamera.hevc", dest / "fcamera.hevc")
    shutil.copy2(directory / "rlog.zst", dest / "rlog.zst")
    old_inode = (dest / "fcamera.hevc").stat().st_ino
    code, out, err = run(capfd)
    assert code == 0, err
    rows = json.loads(out)["files"]
    assert [(row["status"], row["link_status"]) for row in rows] == [
        ("existing", "moved_to_shared"), ("existing", "moved_to_shared"),
    ]
    assert_linked(root, videos, 0, "fcamera.hevc")
    assert_linked(root, videos, 0, "rlog.zst")
    assert shared_file(videos, 0, "fcamera.hevc").stat().st_ino == old_inode  # moved, not copied
    assert rows[0]["shared_path"] == str(shared_file(videos, 0, "fcamera.hevc"))


def test_old_regular_file_identical_to_shared_copy_becomes_a_link(source, capfd):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    shutil.copy2(directory / "fcamera.hevc", dest / "fcamera.hevc")
    shared_file(videos, 0, "fcamera.hevc").parent.mkdir(parents=True)
    shutil.copy2(directory / "fcamera.hevc", shared_file(videos, 0, "fcamera.hevc"))
    code, out, err = run(capfd)
    assert code == 0, err
    rows = json.loads(out)["files"]
    assert rows[0]["link_status"] == "replaced_duplicate"
    assert rows[1]["link_status"] == "created"
    assert_linked(root, videos, 0, "fcamera.hevc")


def test_old_regular_file_with_other_size_than_shared_copy_is_refused(source, capfd):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    (dest / "fcamera.hevc").write_bytes(b"older and longer recording")
    shared = shared_file(videos, 0, "fcamera.hevc")
    shared.parent.mkdir(parents=True)
    shutil.copy2(directory / "fcamera.hevc", shared)
    code, out, _ = run(capfd)
    assert code == 1
    report = json.loads(out)
    assert report["ok"] is False
    assert "size differs" in report["files"][0]["error"]
    assert str(shared) in report["files"][0]["error"]
    assert [row["status"] for row in report["files"]] == ["failed", "planned"]
    assert not (dest / "fcamera.hevc").is_symlink()
    assert (dest / "fcamera.hevc").read_bytes() == b"older and longer recording"
    assert shared.read_bytes() == b"0:fcamera.hevc"


def test_existing_different_content_is_not_overwritten_even_if_same_size(source, capfd):
    remote, root, _, videos = source
    segment(remote, 0)
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    original = b"x" * len(b"0:fcamera.hevc")
    (dest / "fcamera.hevc").write_bytes(original)
    code, out, err = run(capfd)
    assert code != 0
    assert "conflict" in out + err
    # The old local file was only moved into the shared folder, never overwritten.
    assert (dest / "fcamera.hevc").read_bytes() == original
    assert shared_file(videos, 0, "fcamera.hevc").read_bytes() == original


def test_different_shared_copy_is_not_overwritten(source, capfd):
    remote, root, _, videos = source
    segment(remote, 0)
    shared = shared_file(videos, 0, "fcamera.hevc")
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"y" * len(b"0:fcamera.hevc"))
    code, out, _ = run(capfd)
    assert code == 1
    assert "content differs" in json.loads(out)["files"][0]["error"]
    assert shared.read_bytes() == b"y" * len(b"0:fcamera.hevc")
    assert not raw_file(root, 0, "fcamera.hevc").exists()


def test_raw_routes_symlink_to_another_file_is_a_conflict(source, tmp_path, capfd):
    remote, root, _, _ = source
    segment(remote, 0)
    other = tmp_path / "other.hevc"
    other.write_bytes(b"other")
    dest = root / "raw/routes" / ROUTE / "0"
    dest.mkdir(parents=True)
    (dest / "fcamera.hevc").symlink_to(other)
    code, out, _ = run(capfd)
    assert code == 1
    assert "points to" in json.loads(out)["files"][0]["error"]
    assert os.readlink(dest / "fcamera.hevc") == str(other)
    assert other.read_bytes() == b"other"


def test_resume_from_rsync_partial_directory(source, capfd):
    remote, root, _, videos = source
    directory = segment(remote, 0)
    payload = bytes(range(256)) * 4096
    (directory / "fcamera.hevc").write_bytes(payload)
    partial = videos / f"{ROUTE}--0" / ".nnslr-partial"
    partial.mkdir(parents=True)
    (partial / "fcamera.hevc").write_bytes(payload[:len(payload) // 2])
    code, out, err = run(capfd)
    assert code == 0, err
    assert (partial.parent / "fcamera.hevc").read_bytes() == payload
    assert raw_file(root, 0, "fcamera.hevc").read_bytes() == payload
    frame = next(row for row in json.loads(out)["files"] if row["name"] == "fcamera.hevc")
    assert frame["status"] == "resumed"
    assert frame["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("args", [
    ["--host", "-oProxyCommand=bad"], ["--host", "user@host;touch bad"],
    ["--route", "../other"], ["--segments", "3-1"], ["--segments", "-1"],
    ["--segments", ""], ["--segments", "1,,2"],
])
def test_invalid_input_is_rejected_before_ssh(source, capfd, args):
    _, root, commands, videos = source
    code, _, _ = run(capfd, *args)
    assert code != 0
    assert not commands.exists()
    assert not root.exists()
    assert not videos.exists()


def test_videos_dir_inside_raw_routes_is_rejected_before_ssh(source, capfd):
    _, root, commands, _ = source
    code, _, err = run(capfd, "--videos-dir", str(root / "raw/routes/videos"))
    assert code == 2
    assert "COMMA_VIDEOS_DIR" in err
    assert not commands.exists()


def test_destination_symlink_escape_is_rejected(source, tmp_path, capfd):
    remote, root, _, videos = source
    segment(remote, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "raw/routes").mkdir(parents=True)
    (root / "raw/routes" / ROUTE).symlink_to(outside, target_is_directory=True)
    code, _, _ = run(capfd)
    assert code != 0
    assert list(outside.iterdir()) == []
    assert not videos.exists()


def test_ssh_failure_is_actionable_and_not_success(source, capfd, monkeypatch):
    monkeypatch.setenv("TEST_SSH_FAIL", "1")
    code, out, err = run(capfd)
    assert code != 0
    assert "ssh" in (out + err).lower()
    assert "Traceback" not in err


def test_host_preference_can_be_saved_and_explicit_host_wins(source, capfd, monkeypatch):
    remote, _, _, _ = source
    segment(remote, 0)
    monkeypatch.setenv("NNSLR_COMMA_HOST", "my-comma")
    code, out, err = run(capfd, "--dry-run")
    assert code == 0, err
    assert json.loads(out)["host"] == "my-comma"
    code, out, err = run(capfd, "--dry-run", "--host", "comma@192.0.2.20")
    assert code == 0, err
    assert json.loads(out)["host"] == "comma@192.0.2.20"


def test_failed_transfer_reports_pending_files_and_rerun_succeeds(source, capfd, monkeypatch):
    remote, root, _, videos = source
    segment(remote, 0)
    monkeypatch.setenv("TEST_TRANSFER_FAIL", "1")
    code, out, _ = run(capfd)
    assert code == 1
    report = json.loads(out)
    assert report["ok"] is False
    assert [row["status"] for row in report["files"]] == ["failed", "planned"]
    assert "rsync_transfer_failed" in report["files"][0]["error"]
    assert not raw_file(root, 0, "fcamera.hevc").exists()
    assert not raw_file(root, 0, "fcamera.hevc").is_symlink()
    assert not shared_file(videos, 0, "fcamera.hevc").exists()
    monkeypatch.delenv("TEST_TRANSFER_FAIL")
    code, out, err = run(capfd)
    assert code == 0, err
    assert json.loads(out)["ok"] is True


def test_partial_directory_symlink_cannot_redirect_resume(source, tmp_path, capfd):
    remote, _, _, videos = source
    segment(remote, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = videos / f"{ROUTE}--0"
    destination.mkdir(parents=True)
    (destination / ".nnslr-partial").symlink_to(outside, target_is_directory=True)
    code, out, err = run(capfd)
    assert code == 2
    assert "destination_symlink" in out + err
    assert list(outside.iterdir()) == []


def test_symlinked_shared_segment_folder_is_rejected(source, tmp_path, capfd):
    remote, _, _, videos = source
    segment(remote, 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    videos.mkdir()
    (videos / f"{ROUTE}--0").symlink_to(outside, target_is_directory=True)
    code, out, err = run(capfd)
    assert code == 2
    assert "destination_symlink" in out + err
    assert list(outside.iterdir()) == []


def test_unknown_route_is_not_reported_as_success(source, capfd):
    _, root, _, videos = source
    code, out, err = run(capfd, "--dry-run")
    assert code == 2
    assert "route_not_found" in out + err
    assert not root.exists()
    assert not videos.exists()
# [nnslr-sync] - END
