// Scales and ticks of the training charts: clean round numbers, epochs 1..planned.

/** A round step near span / count: 1, 2, 2.5 or 5 times a power of ten. */
export function niceStep(span: number, count: number): number {
  if (!(span > 0)) return 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const nice = unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10;
  return nice * power;
}

/** 0 .. a round maximum above max, with its ticks. */
export function valueTicks(max: number, count = 4): {top: number; ticks: number[]} {
  const step = niceStep(max > 0 ? max * 1.05 : 1, count);
  const top = Math.max(step, Math.ceil((max * 1.02) / step) * step);
  const ticks: number[] = [];
  for (let value = 0; value <= top + step / 1000; value += step) ticks.push(Number(value.toFixed(10)));
  return {top, ticks};
}

/** Epoch ticks: 1, then every step (multiples of the step from 5 up, 1 + k * step below), and the last epoch. */
export function epochTicks(epochs: number, room: number): number[] {
  const most = Math.max(2, Math.floor(room / 34));
  if (epochs <= most) return Array.from({length: epochs}, (_, index) => index + 1);
  const step = Math.max(1, Math.ceil(niceStep(epochs, most - 1)));
  const ticks = [1];
  for (let epoch = step >= 5 ? step : 1 + step; epoch < epochs; epoch += step) {
    if (epoch - (ticks[ticks.length - 1] ?? 0) >= step * 0.6) ticks.push(epoch);
  }
  if (ticks.length > 1 && epochs - (ticks[ticks.length - 1] ?? 0) < step * 0.6) ticks.pop();
  ticks.push(epochs);
  return ticks;
}
