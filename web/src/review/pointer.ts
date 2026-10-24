// Mouse and touch on the photo and on the zoom: draw a new box (drawing mode, N), select a box, move the
// corners of the selected one.
import {datasetOf, svgById} from './dom';
import {addManualBox, changed, revealEditor} from './edits';
import {currentRow, selectedDetection, workOf} from './frames';
import {renderStage, renderZoom} from './overlay';
import {render, state} from './state';
import type {Box, Corner, FrameRow, Point} from './types';

function isCorner(value: string | undefined): value is Corner {
  return value === 'nw' || value === 'ne' || value === 'sw' || value === 'se';
}

/** The corner handle under the pointer, if any. */
function cornerOf(target: EventTarget | null): Corner | null {
  const corner = datasetOf(target)?.['corner'];
  return isCorner(corner) ? corner : null;
}

/** The point of the image under the pointer, inside the image. An SVGPoint (single precision), as the boxes
 * drawn so far: a DOMPoint would give the same box with more decimals. */
function toImagePoint(svg: SVGSVGElement, event: PointerEvent, row: FrameRow): Point {
  const matrix = svg.getScreenCTM();
  let point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  if (matrix !== null) point = point.matrixTransform(matrix.inverse());
  return [Math.min(Math.max(point.x, 0), row.width), Math.min(Math.max(point.y, 0), row.height)];
}

function drawDraft(a: Point, b: Point): void {
  const draft = svgById('draft');
  draft.setAttribute('x', String(Math.min(a[0], b[0])));
  draft.setAttribute('y', String(Math.min(a[1], b[1])));
  draft.setAttribute('width', String(Math.abs(a[0] - b[0])));
  draft.setAttribute('height', String(Math.abs(a[1] - b[1])));
}

function onPointerDown(event: PointerEvent): void {
  const row = currentRow();
  const svg = event.currentTarget;
  if (!row || !(svg instanceof SVGSVGElement)) return;
  if (svg.id === 'scene' && state.drawing) {
    event.preventDefault();
    const start = toImagePoint(svg, event, row);
    state.drag = {kind: 'draw', svg, start};
    svg.setPointerCapture(event.pointerId);
    svgById('draft').setAttribute('visibility', 'visible');
    drawDraft(start, start);
    return;
  }
  const corner = cornerOf(event.target);
  if (corner !== null) {
    event.preventDefault();
    state.drag = {kind: 'resize', svg, corner, moved: false};
    svg.setPointerCapture(event.pointerId);
    svg.classList.add('dragging');
  }
}

function onPointerMove(event: PointerEvent): void {
  const drag = state.drag;
  if (!drag || event.currentTarget !== drag.svg) return;
  const row = currentRow();
  if (!row) return;
  const p = toImagePoint(drag.svg, event, row);
  if (drag.kind === 'draw') {
    drawDraft(drag.start, p);
    return;
  }
  const d = selectedDetection();
  if (!d) return;
  const b: Box = [...d.bbox_xyxy];
  if (drag.corner === 'nw') { b[0] = p[0]; b[1] = p[1]; }
  if (drag.corner === 'ne') { b[2] = p[0]; b[1] = p[1]; }
  if (drag.corner === 'sw') { b[0] = p[0]; b[3] = p[1]; }
  if (drag.corner === 'se') { b[2] = p[0]; b[3] = p[1]; }
  const x1 = Math.min(b[0], b[2]);
  const x2 = Math.max(b[0], b[2]);
  const y1 = Math.min(b[1], b[3]);
  const y2 = Math.max(b[1], b[3]);
  if (x2 - x1 >= 2 && y2 - y1 >= 2) {
    d.bbox_xyxy = [x1, y1, x2, y2];
    drag.moved = true;
    renderStage(row, workOf(row));
    renderZoom(row, d);
  }
}

function finishDrag(event: PointerEvent): void {
  const row = currentRow();
  const drag = state.drag;
  if (!row || !drag || event.currentTarget !== drag.svg) return;
  state.drag = null;
  drag.svg.classList.remove('dragging');
  // The click that follows a drag must not select or deselect anything.
  state.ignoreClickUntil = Date.now() + 400;
  if (drag.kind === 'draw') {
    svgById('draft').setAttribute('visibility', 'hidden');
    if (event.type === 'pointercancel') return;
    const p = toImagePoint(drag.svg, event, row);
    const box: Box = [
      Math.min(drag.start[0], p[0]), Math.min(drag.start[1], p[1]),
      Math.max(drag.start[0], p[0]), Math.max(drag.start[1], p[1]),
    ];
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
    addManualBox(row, box, 'unreadable', null);
    return;
  }
  if (drag.moved) changed(row);
  else render();
}

// A touch on a corner, or anywhere while drawing, must not scroll the page.
function onTouchStart(event: TouchEvent): void {
  const svg = event.currentTarget;
  if (svg instanceof SVGSVGElement && svg.id === 'scene' && state.drawing) {
    event.preventDefault();
    return;
  }
  if (cornerOf(event.target) !== null) event.preventDefault();
}

/** A click on a box selects it; on the photo elsewhere it clears the selection. */
function onSceneClick(event: MouseEvent): void {
  if (Date.now() < state.ignoreClickUntil || state.drawing) return;
  const data = datasetOf(event.target);
  const index = data?.['index'];
  if (index !== undefined) {
    state.selected = Number(index);
    render();
    revealEditor();
    return;
  }
  if (data?.['corner']) return;
  if (state.selected !== -1) {
    state.selected = -1;
    render();
  }
}

export function initPointer(): void {
  for (const svg of [svgById('scene'), svgById('zoom')]) {
    svg.addEventListener('pointerdown', onPointerDown);
    svg.addEventListener('pointermove', onPointerMove);
    svg.addEventListener('pointerup', finishDrag);
    svg.addEventListener('pointercancel', finishDrag);
    svg.addEventListener('touchstart', onTouchStart, {passive: false});
  }
  svgById('scene').addEventListener('click', onSceneClick);
}
