// Numbers and times in Italian.

/** Thousands with a dot, also for four digits (toLocaleString('it-IT') leaves 8340 ungrouped). */
export function thousands(value: number): string {
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, '.');
}

/** "2026-10-23 18:55:09" -> "18:55" today, otherwise "23/10 18:55". */
export function when(started: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/.exec(started);
  if (!match) return started;
  const [, year, month, day, hour, minute] = match;
  const now = new Date();
  const today = Number(year) === now.getFullYear() && Number(month) === now.getMonth() + 1
    && Number(day) === now.getDate();
  return (today ? '' : day + '/' + month + ' ') + hour + ':' + minute;
}

/** The clock now, "07:46:55". */
export function clock(): string {
  return new Date().toLocaleTimeString('it-IT');
}
