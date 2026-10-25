// The icons of the site, hand-made line drawings on a 24 × 24 grid: one stroke width (1.8, round ends and
// joins), currentColor, so they follow the text colour and both themes. A shape with fill: true is solid
// (play marks, the stop square, dots). Drawn by components/Icon.vue.

export interface Shape {
  d: string;
  fill?: true;
}

/** A dot of radius r at (x, y), as a path. */
const dot = (x: number, y: number, r = 1.1): Shape =>
  ({d: `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0z`, fill: true});
/** A circle outline. */
const ring = (x: number, y: number, r: number): string => `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;

export const ICONS = {
  /** Download from the comma: an arrow down into the device. */
  download: [{d: 'M12 3.5v10.5'}, {d: 'M7.5 9.8 12 14.3l4.5-4.5'}, {d: 'M4 14.5v3.3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.3'}],
  /** Prepare new routes: a film frame, an arrow, the sign it finds. */
  prepare: [{d: 'M3.5 6.5h7a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1z'}, {d: 'M2.5 9.5h9M2.5 14.5h9'},
    {d: 'M13.6 12h3.1'}, {d: 'M15.4 10.5l1.5 1.5-1.5 1.5'}, {d: ring(19.6, 12, 2.7)}],
  /** Import reviews: the dataset (a stack) and a check. */
  import: [{d: 'M2.8 6.2c0-1.3 2.6-2.4 5.8-2.4s5.8 1.1 5.8 2.4-2.6 2.4-5.8 2.4-5.8-1.1-5.8-2.4z'},
    {d: 'M2.8 6.2v11.4c0 1.3 2.6 2.4 5.8 2.4 1.2 0 2.3-.15 3.2-.4'}, {d: 'M14.4 6.2v5'}, {d: 'M2.8 11.9c0 1.3 2.6 2.4 5.8 2.4 1 0 2-.1 2.8-.3'},
    {d: 'M14.3 17.3l2.4 2.4 4.8-5.2'}],
  /** Start training: the GPU chip with a play mark. */
  train: [{d: 'M7.5 5.5h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z'},
    {d: 'M9.5 2.5v3M14.5 2.5v3M9.5 18.5v3M14.5 18.5v3M2.5 9.5h3M2.5 14.5h3M18.5 9.5h3M18.5 14.5h3'},
    {d: 'M10.6 9.6v4.8l4-2.4z', fill: true}],
  /** Update the code: two arrows round. */
  update: [{d: 'M19.5 10.5a7.6 7.6 0 0 0-13.7-3.6L4.5 8.6'}, {d: 'M4.5 4.2v4.4h4.4'},
    {d: 'M4.5 13.5a7.6 7.6 0 0 0 13.7 3.6l1.3-1.7'}, {d: 'M19.5 19.8v-4.4h-4.4'}],
  /** Stop a job. */
  stop: [{d: 'M8 6.5h8a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5H8A1.5 1.5 0 0 1 6.5 16V8A1.5 1.5 0 0 1 8 6.5z', fill: true}],
  /** Details of a job: a page with lines. */
  details: [{d: 'M14 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8z'}, {d: 'M14 3.5V8h4.5'}, {d: 'M8.8 12.5h6.4M8.8 16h6.4'}],
  /** Resume review: play on a round sign. */
  resume: [{d: ring(12, 12, 8.6)}, {d: 'M10.2 8.6v6.8l5.6-3.4z', fill: true}],
  /** All routes: a list. */
  routes: [{d: 'M9 6.5h11M9 12h11M9 17.5h11'}, dot(4.8, 6.5), dot(4.8, 12), dot(4.8, 17.5)],
  /** Starting point: pre-trained on ImageNet (a grid of photos). */
  imagenet: [{d: 'M4.5 3.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zM14.5 3.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zM4.5 13.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1zM14.5 13.5h5a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z'},
    {d: 'M3.8 9.6l2.4-2.6 1.6 1.6 1-1 1.6 1.8'}, {d: ring(17, 17, 2.1)}, dot(17.6, 6.2, .9)],
  /** Starting point: random weights (a die). */
  random: [{d: 'M6.5 3.5h11a3 3 0 0 1 3 3v11a3 3 0 0 1-3 3h-11a3 3 0 0 1-3-3v-11a3 3 0 0 1 3-3z'},
    dot(8.4, 8.4, 1.3), dot(15.6, 8.4, 1.3), dot(12, 12, 1.3), dot(8.4, 15.6, 1.3), dot(15.6, 15.6, 1.3)],
  /** Starting point: the stable model (a badge with a star). */
  stable: [{d: ring(12, 9.5, 6.3)}, {d: 'M8.6 14.8 7.5 21l4.5-2.4 4.5 2.4-1.1-6.2'},
    {d: 'M12 6.4l1 2 2.2.3-1.6 1.5.4 2.2-2-1-2 1 .4-2.2-1.6-1.5 2.2-.3z'}],
  /** Apply previous: the frame before behind this one, the sign carried over. */
  previous: [{d: 'M13.5 8.5V5A1.5 1.5 0 0 0 12 3.5H5A1.5 1.5 0 0 0 3.5 5v7A1.5 1.5 0 0 0 5 13.5h3.5'},
    {d: 'M11 9.5h8a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 9.5 19v-8A1.5 1.5 0 0 1 11 9.5z'},
    {d: ring(15, 15, 2.5)}, {d: ring(8.5, 8.5, 1.6)}],
  /** The AI reviewer: a large and a small sparkle. */
  ai: [{d: 'M10 3.5c.6 3.7 2.3 5.4 6 6-3.7.6-5.4 2.3-6 6-.6-3.7-2.3-5.4-6-6 3.7-.6 5.4-2.3 6-6z'},
    {d: 'M17.5 14c.3 1.9 1.1 2.7 3 3-1.9.3-2.7 1.1-3 3-.3-1.9-1.1-2.7-3-3 1.9-.3 2.7-1.1 3-3z'}],
  /** Starting point: an earlier run (a clock turning back). */
  history: [{d: 'M3.8 12a8.2 8.2 0 1 0 2.4-5.8L4 8.4'}, {d: 'M4 4.2v4.2h4.2'}, {d: 'M12 7.8V12l3 2'}],
} satisfies Record<string, Shape[]>;

export type IconName = keyof typeof ICONS;
