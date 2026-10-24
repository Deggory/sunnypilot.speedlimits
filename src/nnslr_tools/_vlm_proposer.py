"""The proposals of a route made by Qwen3-VL-8B-Instruct, a model that sees images (nnslr preannotate --proposer
qwen; installed by scripts/setup_vlm.sh). Each frame is shown whole with one question: every traffic sign, as JSON,
with its box, its kind and the number on it. The answer becomes proposals in the same rows as
_preannotate_worker writes. Run with the GPU environment (.venv-gpu) on the request.json written by
preannotate.run_preannotation. The model gives no score: the proposals carry none."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

PROMPT = ('This is a frame of a dashcam in a car driving in Switzerland. Find every traffic sign in it: speed '
          'limits, end of a limit, zone signs, electronic signs over the road, and any other road sign, also small '
          'and far ones. Answer only with JSON, a list of objects {"bbox_2d": [x1, y1, x2, y2], "kind": one of '
          '"speed_limit", "end_of_limit", "zone", "variable_display", "other_sign", "speed": the number on the '
          'sign, or null}. Coordinates are relative, from 0 to 1000. With no sign, answer [].')
# The kinds of the answer, as the families of the review page.
KINDS = {'speed_limit': 'maximum_speed', 'end_of_limit': 'cancellation', 'zone': 'zone',
         'variable_display': 'variable_display', 'other_sign': 'other_sign'}
# Qwen3-VL gives boxes relative to the image, from 0 to 1000.
SCALE = 1000.0


def parse_answer(text: str, width: int, height: int) -> list[dict]:
    """The proposals of an answer: the JSON list in it (inside ``` fences or not), boxes in frame pixels; items
    without a usable box are dropped, an unknown kind is another sign, a speed only when plausible."""
    match = re.search(r'\[.*\]', text, re.S)
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    found = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        box = item.get('bbox_2d') or item.get('bbox')
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = sorted((max(0.0, min(SCALE, x1)), max(0.0, min(SCALE, x2))))
        y1, y2 = sorted((max(0.0, min(SCALE, y1)), max(0.0, min(SCALE, y2))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        kind = item.get('kind') if item.get('kind') in KINDS else 'other_sign'
        family = KINDS[kind]
        speed = item.get('speed')
        value = int(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) and 5 <= speed <= 150 \
            and family in ('maximum_speed', 'zone', 'variable_display') else None
        found.append({'bbox_xyxy': [round(x1 * width / SCALE, 1), round(y1 * height / SCALE, 1),
                                    round(x2 * width / SCALE, 1), round(y2 * height / SCALE, 1)],
                      'class_id': None, 'model_label': 'qwen3-vl:' + kind, 'score': None, 'family': family,
                      'value_kph': value, 'value_state': 'proposed' if value is not None else 'unknown',
                      'review_state': 'pending', 'applicability': 'unresolved', 'supplementary_panel': 'unknown'})
    return found


def run(request: dict) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    root = Path(request['data_root'])
    model_dir = request['model']['path']
    # The V100 has no bfloat16; sdpa attention (no flash-attention on sm_70).
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float16, device_map='cuda',
                                                        attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(model_dir)
    print(f"Proposals with {request['model']['repository']} on the GPU", file=sys.stderr, flush=True)
    output = Path(request['output'])
    (output / 'images').mkdir()
    frames = request['frames']
    with (output / 'preannotations.jsonl').open('w') as manifest:
        for ordinal, frame in enumerate(frames):
            with Image.open(root / frame['image_path']) as raw:
                image = raw.convert('RGB')
            width, height = image.size
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors='pt').to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=600, do_sample=False)
            answer = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            detections = parse_answer(answer, width, height)
            for i, item in enumerate(detections):
                item['detection_id'] = f"{frame['frame_key']}/{i}"
            preview = f'images/{ordinal:08}.jpg'
            image.save(output / preview, quality=92)
            row = {**frame, 'schema_version': 1, 'kind': 'preannotated_frame', 'run_id': request['run_id'],
                   'width': width, 'height': height, 'preview_path': preview, 'detections': detections,
                   'review_state': 'pending', 'ground_truth': False, 'model': request['model'], 'road_model': None,
                   'ocr': None, 'vlm_answer': answer[:4000],
                   'inference': {'provider': 'CUDAExecutionProvider', 'confidence': None, 'nms_iou': None, 'tiles': 1}}
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
