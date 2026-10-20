# Recovery audit — 2026-10-21

## Scope and sources

Destination: `cristianku/sunnypilot.speedlimits`.

| Source | Frozen commit | Recovery |
| --- | --- | --- |
| `cristianku/sunnypilot:nn-speed-limit-vision` | `975fdde8ce767f4e1fc7129a92b3e13589857cbf` | All 32 files changed since its merge base, plus three unchanged test-context files, stored as an isolated partial snapshot |
| `cristianku/openpilot-nnslr-tools:main` | `82a603f4bfdd2dd661e557af509b9addde3c9c18` | 99 source/test/config/documentation files plus the original README |

The runtime's vendored-core manifest identifies the older training-source commit `1a212ee72aa8ba7fff89559091095c2274e19887`. Following that explicit reference revealed the more complete training repository. Its current source was recovered separately; the old runtime snapshot was not silently updated to use the newer core.

All original Git blob hashes were checked during extraction. `provenance/import.json` maps original paths and hashes to their new destinations and records the small repository-identity changes.

## Reusable training work

The recovered CLI includes route copying on explicit invocation, frame extraction, video/log alignment, proposal review, annotation validation, leakage-aware frozen splits, detector/reader training, evaluation, error mining, replay, ONNX export and immutable-bundle verification.

The detector baseline is `ssdlite320_mobilenet_v3_large`; the crop-reader baseline is `mobilenet_v3_small`. Importing their source is not evidence of completed training or target compatibility.

The portable core includes explicit absent/unreadable values, bounded observations, session-aware timestamps, deterministic validation, temporal consensus, health gating and a CAR/MAP/VISION advisory comparator. It does not supply a verified real-world road-ownership or sign-passage algorithm.

## Runtime status

The source branch is 36 commits ahead and 269 commits behind the inspected sunnypilot master `922513f3128b00611d695ce465989a9bc8af9d59`. Its merge base is `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56`. These counts describe the inspection, not an ongoing compatibility promise.

Useful runtime work includes:
- Owned NV12 frame copies with dimensions/stride and capture identity.
- A bounded latest-frame lifecycle.
- A dedicated `speedVisionState` message and publisher.
- Opt-in process parameters and the `speedvisiond` bootstrap.
- A separate VISION badge and bounded diagnostic-journal utility.
- Source-level tests for the above contracts.

The model lock says `model.state = unavailable`. `load_runner()` always returns `UnavailableRunner`, even if a model is marked ready. The daemon only publishes unavailable states; it does not yet consume camera frames for inference. The diagnostics utility is present but is not wired into that daemon.

The source explicitly excludes `SpeedLimitResolver`, `SpeedLimitAssist` and vehicle-control consumers. Therefore it does not deliver the requested operational speed-limit integration yet.

The partial snapshot is retained for a deliberate port against the current target branch. Replacing current cereal/UI/process files wholesale with these historical copies is not an installation method.

The old VISION widget checks whether a message has ever been seen, without checking message liveness or freshness. A future runtime port must invalidate stale display state when its producer stops.

## Changes made during recovery

- Updated package homepage and new core-export provenance to `cristianku/sunnypilot.speedlimits`, including the associated assertion.
- Preserved the frozen runtime's original hashes and old source manifest.
- Added current repository documentation, attribution, per-file provenance and a runtime snapshot test job.
- Preserved old task plans/guides as historical source material; the current development plan is `docs/development-plan.md`.

No model weights, vehicle recordings, GPU workloads, device connections or changes to either source repository are part of this recovery.
