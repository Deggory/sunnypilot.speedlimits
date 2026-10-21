"""Readers accept raw/routes symlinks into the shared videos folder, and nothing else."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools import cli
from nnslr_tools.manifest import NnslerManifestError, SegmentStatus, normalize_relpath
from nnslr_tools.route_io import build_route_manifest
from nnslr_tools.sync import link_videos

ROUTE = "000001a3--c20ba54385"


@pytest.fixture
def linked(tmp_path, monkeypatch):
    videos = tmp_path / "comma videos"
    root = tmp_path / "data root"
    segment = videos / f"{ROUTE}--0"
    segment.mkdir(parents=True)
    (segment / "fcamera.hevc").write_bytes(b"video0")
    (segment / "rlog.zst").write_bytes(b"log0")
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(videos))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(root))
    report = link_videos(route=ROUTE)
    assert report["ok"] is True
    return videos, root


def test_route_manifest_accepts_links_into_shared_folder(linked, tmp_path, capsys):
    _, root = linked
    manifest = build_route_manifest(root, Path("raw/routes") / ROUTE)
    assert manifest.segment_status[0] == SegmentStatus.COMPLETE
    rows = {row.relpath: row for row in manifest.files}
    assert sorted(rows) == [f"raw/routes/{ROUTE}/0/fcamera.hevc", f"raw/routes/{ROUTE}/0/rlog.zst"]
    video = rows[f"raw/routes/{ROUTE}/0/fcamera.hevc"]
    assert video.sha256 == hashlib.sha256(b"video0").hexdigest()
    assert video.size_bytes == len(b"video0")
    output = tmp_path / "routes.jsonl"
    assert cli.main(["route-manifest", f"raw/routes/{ROUTE}", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["file_count"] == 2


def test_normalize_relpath_accepts_only_file_links_into_shared_folder(linked, tmp_path):
    videos, root = linked
    relpath = f"raw/routes/{ROUTE}/0/fcamera.hevc"
    assert normalize_relpath(root, Path(relpath)) == relpath

    # A link out of raw/routes to anything else outside the data root.
    outside = tmp_path / "outside.hevc"
    outside.write_bytes(b"outside")
    (root / "raw/routes" / ROUTE / "0/ecamera.hevc").symlink_to(outside)
    with pytest.raises(NnslerManifestError) as error:
        normalize_relpath(root, Path(f"raw/routes/{ROUTE}/0/ecamera.hevc"))
    assert error.value.reason == "path_traversal"
    assert "COMMA_VIDEOS_DIR" in str(error.value)

    # A symlinked segment folder may not redirect the segment, even into the shared folder.
    (root / "raw/routes" / ROUTE / "1").symlink_to(videos / f"{ROUTE}--0", target_is_directory=True)
    with pytest.raises(NnslerManifestError):
        normalize_relpath(root, Path(f"raw/routes/{ROUTE}/1/fcamera.hevc"))

    # Derived outputs keep the strict check: no link out of the data root.
    (root / "derived/frames").mkdir(parents=True)
    (root / "derived/frames/frame.png").symlink_to(videos / f"{ROUTE}--0" / "fcamera.hevc")
    with pytest.raises(NnslerManifestError):
        normalize_relpath(root, Path("derived/frames/frame.png"))

    # Traversal stays rejected.
    with pytest.raises(NnslerManifestError):
        normalize_relpath(root, Path(f"raw/routes/{ROUTE}/0/../../../../../outside.hevc"))


def test_links_are_rejected_when_the_shared_folder_is_different(linked, tmp_path, monkeypatch):
    _, root = linked
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(tmp_path / "another folder"))
    with pytest.raises(NnslerManifestError) as error:
        normalize_relpath(root, Path(f"raw/routes/{ROUTE}/0/fcamera.hevc"))
    assert "COMMA_VIDEOS_DIR" in str(error.value)


def test_extract_frames_route_reads_linked_video(tmp_path, monkeypatch, capsys):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe needed")
    videos = tmp_path / "comma videos"
    root = tmp_path / "data root"
    segment = videos / f"{ROUTE}--0"
    segment.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x48:r=20:d=0.1",
                    "-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:log-level=error",
                    str(segment / "fcamera.hevc")], check=True, capture_output=True)
    (segment / "rlog.zst").write_bytes(b"log0")
    monkeypatch.setenv("COMMA_VIDEOS_DIR", str(videos))
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(root))
    assert link_videos(route=ROUTE)["ok"] is True
    shared_before = sorted(str(p.relative_to(videos)) for p in videos.rglob("*"))

    code = cli.main(["extract-frames", "--route", ROUTE])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert json.loads(captured.out)["frame_count"] == 1
    folder = root / "derived/frames" / ROUTE / "0"
    assert (folder / "frame_00000000.png").read_bytes().startswith(b"\x89PNG")
    row = json.loads((folder / "frames.jsonl").read_text())
    assert row["source_video"] == str(root / "raw/routes" / ROUTE / "0/fcamera.hevc")
    # Outputs stay in the data root; nothing is written into the shared folder.
    assert sorted(str(p.relative_to(videos)) for p in videos.rglob("*")) == shared_before
