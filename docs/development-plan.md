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

## 5. Package controlled updates

- [ ] Create a deployment skill/script that accepts an exact target branch and immutable model version.
- [ ] Verify artifact hashes and compatibility; prepare a reviewable branch/PR.
- [ ] Keep a documented rollback to the prior model and adapter.

This plan describes future implementation. The recovered advisory-only runtime and historical plans are reference material, not completed SLA support.
