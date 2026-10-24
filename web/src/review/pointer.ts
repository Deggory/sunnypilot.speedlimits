// Mouse and touch on the photo and on the zoom: draw a new box (drawing mode, N), select a box, move the
// corners of the selected one. The handlers of both pictures (Stage.vue, BoxEditor.vue); the point under the
// pointer is the SVG's own math.
import {nextTick} from 'vue';
import {datasetOf} from './dom';
import {addManualBox, changed} from './edits';
import {currentRow, selectedDetection} from './frames';
import {revealEditor} from './scroll';
import {state} from './store';
import type {Box, Corner, FrameRow, Point, Surface} from './types';

function isCorner(value: string | undefined): value is Corner {
  return value === 'nw' || value === 'ne' || value === 'sw' || value === 'se';
}

/** The corner handle under the pointer, if any. */
function cornerOf(target: EventTarget | null): Corner | null {
  const corner = datasetOf(target)?.['corner'];
  return isCorner(corner) ? corner : null;
}

/** The picture the handler is on (the listeners are on the <svg> elements). */
function svgOf(event: Event): SVGSVGElement | null {
  return event.currentTarget instanceof SVGSVGElement ? event.currentTarget : null;
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
  state.draft = {
    x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]), width: Math.abs(a[0] - b[0]), height: Math.abs(a[1] - b[1]),
  };
}

export function onPointerDown(event: PointerEvent, surface: Surface): void {
  const row = currentRow();
  const svg = svgOf(event);
  if (!row || svg === null) return;
  if (surface === 'scene' && state.drawing) {
    event.preventDefault();
    const start = toImagePoint(svg, event, row);
    state.drag = {kind: 'draw', svg: surface, start};
    svg.setPointerCapture(event.pointerId);
    drawDraft(start, start);
    return;
  }
  const corner = cornerOf(event.target);
  if (corner !== null) {
    event.preventDefault();
    state.drag = {kind: 'resize', svg: surface, corner, moved: false};
    svg.setPointerCapture(event.pointerId);
  }
}

export function onPointerMove(event: PointerEvent, surface: Surface): void {
  const drag = state.drag;
  const svg = svgOf(event);
  if (!drag || drag.svg !== surface || svg === null) return;
  const row = currentRow();
  if (!row) return;
  const p = toImagePoint(svg, event, row);
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
  }
}

export function finishDrag(event: PointerEvent, surface: Surface): void {
  const row = currentRow();
  const drag = state.drag;
  const svg = svgOf(event);
  if (!row || !drag || drag.svg !== surface || svg === null) return;
  state.drag = null;
  // The click that follows a drag must not select or deselect anything.
  state.ignoreClickUntil = Date.now() + 400;
  if (drag.kind === 'draw') {
    state.draft = null;
    if (event.type === 'pointercancel') return;
    const p = toImagePoint(svg, event, row);
    const box: Box = [
      Math.min(drag.start[0], p[0]), Math.min(drag.start[1], p[1]),
      Math.max(drag.start[0], p[0]), Math.max(drag.start[1], p[1]),
    ];
    if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
    addManualBox(row, box, 'unreadable', null);
    return;
  }
  if (drag.moved) changed(row);
}

// A touch on a corner, or anywhere while drawing, must not scroll the page.
export function onTouchStart(event: TouchEvent, surface: Surface): void {
  if (surface === 'scene' && state.drawing) {
    event.preventDefault();
    return;
  }
  if (cornerOf(event.target) !== null) event.preventDefault();
}

/** A click on a box selects it; on the photo elsewhere it clears the selection. */
export function onSceneClick(event: MouseEvent): void {
  if (Date.now() < state.ignoreClickUntil || state.drawing) return;
  const data = datasetOf(event.target);
  const index = data?.['index'];
  if (index !== undefined) {
    state.selected = Number(index);
    void nextTick(revealEditor);
    return;
  }
  if (data?.['corner']) return;
  if (state.selected !== -1) state.selected = -1;
}
