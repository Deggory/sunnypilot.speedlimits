// The photo and what is drawn on it: the boxes with their dark under-stroke and labels, the corner handles of
// the selected box, the boxes of the second reading, and the zoom of the selected box in the panel.
import {byId} from '../common/dom';
import {boxColor, hintLabel, shortName} from './catalog';
import {coarsePointer, svgById, svgNode} from './dom';
import {currentRow} from './frames';
import {boxArea, boxesOverlap, freeLabelY, labelWidth, rectOf, type PlacedLabel} from './geometry';
import {hintsOf, state} from './state';
import type {Box, Corner, Detection, FrameEdit, FrameRow} from './types';

/** On-screen pixels per image pixel of an SVG showing width x height. */
function sceneScale(svg: SVGSVGElement, width: number, height: number): number {
  const rect = svg.getBoundingClientRect();
  if (!rect.width || !rect.height || !width || !height) return 1;
  return Math.min(rect.width / width, rect.height / height);
}

/** Boxes and hints are hidden until the photo of their frame has loaded. */
function showLayers(visible: boolean): void {
  const visibility = visible ? '' : 'hidden';
  svgById('overlays').style.visibility = visibility;
  svgById('hintLayer').style.visibility = visibility;
}

export function renderStage(row: FrameRow, work: FrameEdit): void {
  const scene = svgById<SVGSVGElement>('scene');
  scene.setAttribute('viewBox', '0 0 ' + row.width + ' ' + row.height);
  const photo = svgById('photo');
  if (photo.getAttribute('href') !== row.preview_path) {
    // The boxes of the new frame wait for its photo: never drawn on the previous one.
    photo.setAttribute('href', row.preview_path);
    byId('stage').dataset['photoKey'] = '';
    showLayers(false);
  }
  photo.setAttribute('width', String(row.width));
  photo.setAttribute('height', String(row.height));
  scene.classList.toggle('drawing', state.drawing);
  byId('stageHint').hidden = !state.drawing;

  const k = sceneScale(scene, row.width, row.height);
  const fontSize = 14 / k;
  const placed: PlacedLabel[] = [];
  const handleSize = 8 / k;
  const hitSize = (coarsePointer() ? 36 : 20) / k;
  const overlays = svgById('overlays');
  overlays.replaceChildren();
  // Larger boxes first, so that a small box inside a large one stays clickable.
  const order = work.detections.map((_, index) => index);
  order.sort((a, b) => boxAreaAt(work, b) - boxAreaAt(work, a));
  for (const i of order) {
    const d = work.detections[i];
    if (d === undefined) continue;
    const b = d.bbox_xyxy;
    const isSelected = i === state.selected;
    let cls = 'box';
    if (d.model_label) cls += ' proposal';
    if (isSelected) cls += ' selected';
    const geometry = rectOf(b);
    // A dark under-stroke first: the box reads on snow and sky too (iOS Safari ignores CSS filters on SVG).
    overlays.append(svgNode('rect', {class: cls.replace('box', 'box-under'), ...geometry}));
    const rect = svgNode('rect', {class: cls, 'data-index': i, ...geometry});
    if (!isSelected) rect.style.stroke = boxColor(d);
    overlays.append(rect);
    const text = (i + 1) + ' · ' + shortName(d);
    const width = labelWidth(text, fontSize);
    // The selected box has its corners outside: its label goes above them.
    const top = isSelected ? b[1] - handleSize - fontSize * 0.2 : b[1] - fontSize * 0.35;
    const y = Math.max(fontSize, freeLabelY(placed, b[0], Math.max(fontSize, top), width, fontSize, -fontSize));
    placed.push({x: b[0], y, w: width, h: fontSize, text});
    overlays.append(label('tag', text, b[0], y, fontSize, k));
  }
  renderHintLayer(row, k, placed);
  const selected = work.detections[state.selected];
  if (selected !== undefined) appendHandles(overlays, selected.bbox_xyxy, handleSize, hitSize, true);
}

function boxAreaAt(work: FrameEdit, index: number): number {
  const d = work.detections[index];
  return d ? boxArea(d.bbox_xyxy) : 0;
}

function label(cls: string, text: string, x: number, y: number, fontSize: number, k: number): SVGTextElement {
  const tag = svgNode('text', {
    x, y, class: cls, 'font-size': fontSize, 'stroke-width': 3.5 / k, 'paint-order': 'stroke',
  });
  tag.textContent = text;
  return tag;
}

const CORNERS: readonly Corner[] = ['nw', 'ne', 'sw', 'se'];

function cornerPoint(b: Box, corner: Corner): [number, number] {
  const x = corner === 'nw' || corner === 'sw' ? b[0] : b[2];
  const y = corner === 'nw' || corner === 'ne' ? b[1] : b[3];
  return [x, y];
}

// On the photo the corners sit outside the box, so they never cover a small sign.
function appendHandles(layer: SVGElement, b: Box, size: number, hit: number, outside: boolean): void {
  for (const corner of CORNERS) {
    const [x, y] = cornerPoint(b, corner);
    let hx = x - size / 2;
    let hy = y - size / 2;
    if (outside) {
      hx = corner === 'nw' || corner === 'sw' ? x - size : x;
      hy = corner === 'nw' || corner === 'ne' ? y - size : y;
    }
    layer.append(svgNode('rect', {x: hx, y: hy, width: size, height: size, class: 'handle'}));
  }
  for (const corner of CORNERS) {
    const [x, y] = cornerPoint(b, corner);
    layer.append(svgNode('rect', {
      x: x - hit / 2, y: y - hit / 2, width: hit, height: hit, class: 'handle-hit ' + corner, 'data-corner': corner,
    }));
  }
}

function renderHintLayer(row: FrameRow, k: number, placed: PlacedLabel[]): void {
  const layer = svgById('hintLayer');
  layer.replaceChildren();
  if (!state.showHints) return;
  const fontSize = 12 / k;
  for (const h of hintsOf(row.frame_key)) {
    const b = h.box;
    const geometry = rectOf(b);
    layer.append(svgNode('rect', {class: 'hint-under', ...geometry}));
    layer.append(svgNode('rect', {class: 'hint-box ' + h.confidence, ...geometry}));
    const text = hintLabel(h);
    const width = labelWidth(text, fontSize);
    // The two passes often see the same sign: one label is enough.
    if (placed.some((p) => p.text === text && p.box && boxesOverlap(p.box, b))) continue;
    // Below the box, where the label of a reviewed box never is.
    const y = Math.min(row.height, freeLabelY(placed, b[0], b[3] + fontSize * 1.05, width, fontSize, fontSize));
    placed.push({x: b[0], y, w: width, h: fontSize, text, box: b});
    layer.append(label('hint-tag', text, b[0], y, fontSize, k));
  }
}

export function renderZoom(row: FrameRow, d: Detection): void {
  const zoom = svgById<SVGSVGElement>('zoom');
  const b = d.bbox_xyxy;
  // While a corner of the zoom is dragged its view stays still, so the corner stays under the finger.
  const keepView = state.drag !== null && state.drag.svg === zoom;
  if (!keepView) {
    const w = b[2] - b[0];
    const h = b[3] - b[1];
    const pad = Math.max(w, h) * 0.35 + 4;
    const x = Math.max(0, b[0] - pad);
    const y = Math.max(0, b[1] - pad);
    let vw = w + pad * 2;
    let vh = h + pad * 2;
    if (x + vw > row.width) vw = row.width - x;
    if (y + vh > row.height) vh = row.height - y;
    zoom.setAttribute('viewBox', x + ' ' + y + ' ' + vw + ' ' + vh);
  }
  const zoomPhoto = svgById('zoomPhoto');
  if (zoomPhoto.getAttribute('href') !== row.preview_path) zoomPhoto.setAttribute('href', row.preview_path);
  zoomPhoto.setAttribute('width', String(row.width));
  zoomPhoto.setAttribute('height', String(row.height));
  const zoomBox = svgById('zoomBox');
  for (const [name, value] of Object.entries(rectOf(b))) zoomBox.setAttribute(name, String(value));
  const view = zoom.viewBox.baseVal;
  const kz = view && view.width && view.height ? sceneScale(zoom, view.width, view.height) : 1;
  const layer = svgById('zoomHandles');
  layer.replaceChildren();
  appendHandles(layer, b, 12 / kz, (coarsePointer() ? 36 : 22) / kz, false);
}

/** The photo has loaded (or failed): show its boxes, and say which frame it is (stage[data-photo-key]). */
export function initPhoto(): void {
  const photo = svgById('photo');
  const loaded = (failed: boolean): void => {
    const row = currentRow();
    if (row === null || photo.getAttribute('href') !== row.preview_path) return;
    byId('stage').dataset['photoKey'] = (failed ? 'error:' : '') + row.frame_key;
    showLayers(true);
  };
  photo.addEventListener('load', () => loaded(false));
  photo.addEventListener('error', () => loaded(true));
}
