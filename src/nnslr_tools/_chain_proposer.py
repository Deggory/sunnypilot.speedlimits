"""The proposals of a route made by our own networks (nnslr preannotate --proposer ours): the cutter in use finds the
boxes on the full-resolution frame, the classifier reads each box cut from it, and every box it does not call
"not a sign" becomes a proposal, in the same rows as _preannotate_worker writes. Run with the GPU environment
(.venv-gpu: torch, torchvision) on the request.json written by preannotate.run_preannotation."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def proposal_of(label: str) -> tuple[str, int | None] | None:
    """(family, value) of a classifier label ("maximum_speed:60", "variable_display:unknown", "other_sign",
    "road_marking:30"); None for not_a_sign."""
    if label == 'not_a_sign':
        return None
    family, _, value = label.partition(':')
    family = {'road_marking': 'road_marking_candidate'}.get(family, family)
    return family, int(value) if value.isdigit() else None


def run(request: dict) -> None:
    import torch
    import torchvision.transforms.functional as TF
    from nnslr_tools import chain
    from nnslr_tools.chain import (CLASSIFIER_BATCH, _device, _load_classifier, _load_cutter, cutter_boxes)
    from nnslr_tools.preannotate import suppress_overlaps
    from nnslr_tools.training import crop_rectangle

    root = Path(request['data_root'])
    # The progress of a training lives in logs/progress.json: this worker reports in its log lines only.
    chain.report_progress = lambda *args, **kwargs: None
    device = _device(torch, 'cuda' if torch.cuda.is_available() else 'cpu')
    proposer = request['model']
    cutter, _, _ = _load_cutter({'path': Path(proposer['cutter_path']), 'id': proposer['cutter']}, device)
    classifier, _, classes, transform = _load_classifier({'path': Path(proposer['classifier_path']),
                                                          'id': proposer['classifier']}, device)
    print(f"Proposals with our networks on {device}: cutter {proposer['cutter']}, classifier {proposer['classifier']}",
          file=sys.stderr, flush=True)
    output = Path(request['output'])
    (output / 'images').mkdir()
    frames = request['frames']
    with (output / 'preannotations.jsonl').open('w') as manifest:
        for ordinal, (frame, pixels, boxes) in enumerate(cutter_boxes(cutter, device, frames, root,
                                                                      min_score=request['confidence'],
                                                                      batch_size=4, num_workers=2, task='prepare')):
            height, width = int(pixels.shape[1]), int(pixels.shape[2])
            detections = []
            for start in range(0, len(boxes), CLASSIFIER_BATCH):
                chunk = boxes[start:start + CLASSIFIER_BATCH]
                crops = []
                for item in chunk:
                    left, top, right, bottom = crop_rectangle(item['box'], width, height)
                    crops.append(transform(TF.to_pil_image(pixels[:, top:bottom, left:right])))
                with torch.no_grad():
                    probabilities = torch.softmax(classifier(torch.stack(crops).to(device)).float(), dim=1)
                for item, index, confidence in zip(chunk, probabilities.argmax(1).tolist(),
                                                   probabilities.max(1).values.tolist()):
                    proposal = proposal_of(classes[index])
                    if proposal is None:
                        continue
                    family, value = proposal
                    detections.append({'bbox_xyxy': [round(float(v), 1) for v in item['box']], 'class_id': index,
                                       'model_label': classes[index], 'score': round(float(confidence), 4),
                                       'cutter_score': round(float(item['score']), 4), 'family': family,
                                       'value_kph': value, 'value_state': 'proposed' if value is not None else 'unknown',
                                       'review_state': 'pending', 'applicability': 'unresolved',
                                       'supplementary_panel': 'unknown'})
            detections = suppress_overlaps(detections)
            for i, item in enumerate(detections):
                item['detection_id'] = f"{frame['frame_key']}/{i}"
            preview = f'images/{ordinal:08}.jpg'
            TF.to_pil_image(pixels).save(output / preview, quality=92)
            row = {**frame, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': request['run_id'],
                   'width': width, 'height': height, 'preview_path': preview, 'detections': detections,
                   'review_state': 'pending', 'ground_truth': False, 'model': proposer, 'road_model': None, 'ocr': None,
                   'inference': {'provider': 'CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider',
                                 'confidence': request['confidence'], 'nms_iou': .5, 'tiles': 1}}
            manifest.write(json.dumps(row, allow_nan=False) + '\n')
            manifest.flush()
            print(f"Preannotating {ordinal + 1}/{len(frames)}: segment {frame['segment_index']}, frame "
                  f"{frame['output_index']}, {len(detections)} proposals", file=sys.stderr, flush=True)


if __name__ == '__main__':
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as exc:
        print(f'preannotation worker error: {exc}', file=sys.stderr)
        sys.exit(1)
