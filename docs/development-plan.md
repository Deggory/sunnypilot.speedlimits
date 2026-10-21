# Development plan

## 1. Recover the existing work

- [x] Recover the portable core, training/data pipeline and existing tests.
- [x] Preserve the old runtime integration as a source snapshot with provenance.
- [x] Record that trained weights, target inference and the SLA adapter are missing.

## 2. Establish model evidence

- [ ] Inventory a reviewed speed-limit dataset and its supported sign/value classes.
- [ ] Verify route/site/encounter separation between training, validation and test data.
- [ ] Train and evaluate detector and numeric-reader baselines on the selected training host.
- [ ] Review errors, including adjacent-road signs, zones, cancellations and unreadable/conditional signs.
- [ ] Produce successful ONNX parity exports from the actual trained checkpoints.
- [ ] Freeze a reproducible model bundle with hashes, preprocessing and class mapping.

The existing training commands are the starting point; no checkpoint is claimed by this import.

## 3. Complete and measure target inference

- [ ] Select the actual comma runtime backend after testing model/operator support.
- [ ] Connect camera acquisition, preprocessing, inference and postprocessing.
- [ ] Measure end-to-end latency, dropped frames and impact on the existing driving model.
- [ ] Verify temporal evidence, applicability, passage and invalidation on recorded sequences.
- [ ] Port only the necessary message/process/UI changes to a branch based on the current sunnypilot target.

## 4. Integrate with Speed Limit Assist

- [ ] Add an explicit vision source at the speed-limit selection layer used by SLA.
- [ ] Define source priority, freshness, cancellation and fallback without overwriting CAN provenance.
- [ ] Preserve the distinction between an observed sign and a currently applicable limit.
- [ ] Check units: recognition may carry km/h, while the existing operational speed-limit flow uses m/s.
- [ ] Verify the adapter with recorded input and appropriate target-device checks.

## 4A. Planned Vision / Car / Map selection via SunnyLink (separate gated phase)

**Decision (2026-10-21):** The eventual user-facing Speed Limit Assist source selector must include **Vision** alongside the existing **Car** (vehicle/CAN) and **Map** sources. The selection must be configurable from SunnyLink without requiring a custom SunnyLink frontend.

- [ ] Keep the **initial NNSLR release advisory-only**: `speedVisionState` remains separate from the operational speed-limit resolver and cannot change target speed, braking, or acceleration.
- [ ] After dedicated safety review and measured validation, design a separate **operational Vision source** adapter in the Sunnypilot fork. Do not implement this by writing synthetic vehicle limits to `CarStateSP`, spoofing map data, or modifying OpenDBC for vision inference.
- [ ] Extend the device-generated **SettingsSchema** for the existing speed-limit source/policy setting: expose **Vision Only** and **Vision First**, and provide selection/fallback involving Car and Map. Retain existing Car Only, Map Only, Car First, Map First, Combined and saved values without changing their meanings. Exact additional combinations and fallback precedence must be documented and tested before release.
- [ ] Publish Vision options in the schema **only when the operational feature is available and permitted**. Availability must not be inferred merely from installing a model. The device must continue to expose safe options when Vision is disabled or unavailable.
- [ ] Verify SunnyLink dynamically renders the device-advertised option widget and labels, reads the current selection, saves changes and reloads them correctly (including after device restart). The official frontend already supports device-supplied `SettingsSchema`; propose a frontend PR **only if compatibility testing identifies a real rendering or persistence issue**.
- [ ] Fail closed for Vision limits: use only fresh, validated, applicable **CURRENT** sign hypotheses with verified road ownership/passage; reject uncertain, conditional, cancelled, conflicting, stale and unavailable results. Document the chosen source priority and conservative fallback behavior, explicitly avoiding a stale Vision value becoming a cruise-speed target.
- [ ] Validate unit conversion (km/h to m/s), policy migration, replay tests, hardware behavior, and rollback before seeking upstream Sunnypilot acceptance.

**Boundaries:** `cristianku/sunnypilot.speedlimits` owns training, model artifacts and the source/adapter contract; the Sunnypilot runtime fork owns speed-limit resolver integration and the device-generated SettingsSchema. SunnyLink remains an interface, not the source of speed-limit inference. OpenDBC changes are needed only for genuine vehicle CAN decoding, not to embed the vision model.

This section is a **future proposal**, not an assertion that Vision is currently a working operational SLA source. It explicitly supersedes the *goal* of advisory-only operation only after separate approval and validation; it does not lift the current advisory-only invariants or authorize deployment to a vehicle.

---

## 5. Package controlled updates

- [ ] Create a deployment skill/script that accepts an exact target branch and immutable model version.
- [ ] Verify artifact hashes and compatibility; prepare a reviewable branch/PR.
- [ ] Keep a documented rollback to the prior model and adapter.

This plan describes future implementation. The recovered advisory-only runtime and historical plans are reference material, not completed SLA support.
