# Recovered sunnypilot runtime snapshot

Source: [`nn-speed-limit-vision`](https://github.com/cristianku/sunnypilot/tree/975fdde8ce767f4e1fc7129a92b3e13589857cbf), frozen at `975fdde8ce767f4e1fc7129a92b3e13589857cbf`.

`snapshot/` preserves the complete contents of the 32 files changed by that branch since merge base `a5f44653d7f43ad57fef2f546f3916ec4cbf3c56`, plus three unchanged context files needed by the original isolation tests (resolver, assist and SLA renderer). Original paths are retained under the snapshot root so the recovered source tests can inspect their expected layout.

It is a **partial source snapshot**, not a complete sunnypilot checkout or an installer. It includes old versions of changed cereal, UI and manager files. Port the relevant changes against the actual target; do not overwrite current target files with the snapshot.

## What exists

`openpilot/sunnypilot/speed_vision/` contains the old vendored core, frame adapters, publisher, lifecycle, unavailable-backend contract, daemon bootstrap, diagnostics and their tests. The source also adds the message schema/service, opt-in parameters/process and a separate VISION badge.

## What does not run yet

The artifact lock has no model. The runner has no implemented inference backend. The daemon has no camera/inference loop and publishes unavailable status only. Neither the resolver nor Speed Limit Assist consumes this source.

The original source-level control-isolation tests intentionally enforce its advisory-only behavior. A future operational adapter is a separate, explicit change.

The root `src/speed_vision_core/` is newer than the old core inside the snapshot. Both are retained with distinct source commits; tests and export manifests must not conflate them.

## Run the recovered source tests

From the new repository root, after installing pytest:

```sh
PYTHONPATH="$PWD/integrations/sunnypilot/snapshot" python -m pytest -q \
  --confcutdir=integrations/sunnypilot/snapshot \
  integrations/sunnypilot/snapshot/openpilot/sunnypilot/speed_vision/tests
```

These tests exercise portable contracts and source layout. They do not compile Cap'n Proto, run the UI or validate camera/GPU/device operation.

The original upstream notices are preserved in `LICENSES/`; see the root `NOTICE.md`.
