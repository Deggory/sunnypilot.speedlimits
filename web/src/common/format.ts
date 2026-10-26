// Numbers and times in English, as distill-web shows them: thousands with a comma (11,573), dates en-GB
// ("24 Oct 10:14"), always in the browser's time zone.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad = (value: number): string => String(value).padStart(2, '0');

/** Thousands with a comma, also for four digits: 8579 -> "8,579". */
export function thousands(value: number): string {
  const [whole = '', decimals] = String(value).split('.');
  return whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (decimals === undefined ? '' : '.' + decimals);
}

/** "10:14". */
export function hourMinute(date: Date): string {
  return pad(date.getHours()) + ':' + pad(date.getMinutes());
}

/** "10:14" today, "24 Oct 10:14" another day of this year, "24 Oct 2025 10:14" before. */
export function shortWhen(date: Date): string {
  const now = new Date();
  const sameYear = date.getFullYear() === now.getFullYear();
  if (sameYear && date.getMonth() === now.getMonth() && date.getDate() === now.getDate()) return hourMinute(date);
  return date.getDate() + ' ' + MONTHS[date.getMonth()] + (sameYear ? '' : ' ' + date.getFullYear()) + ' ' + hourMinute(date);
}

/** An ISO time of the server ("2026-10-24T10:14:05+02:00") as shortWhen; '' when missing or unreadable. */
export function isoWhen(iso: string | null | undefined): string {
  if (!iso) return '';
  const parsed = Date.parse(iso);
  return Number.isNaN(parsed) ? '' : shortWhen(new Date(parsed));
}

/** The clock now, "10:14:05". */
export function clock(): string {
  const now = new Date();
  return hourMinute(now) + ':' + pad(now.getSeconds());
}

/** "1 route", "4 routes". */
export function plural(count: number, one: string, many: string = one + 's'): string {
  return thousands(count) + ' ' + (count === 1 ? one : many);
}

/** A reason of the server as a sentence: capital first letter, a full stop at the end. */
export function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return '';
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1) + (/[.!?…]$/.test(trimmed) ? '' : '.');
}

/** The share of frames confirmed, as every page shows it: the nearest whole number, 100 only when all are
 * (nnslr_tools.pages.review_percent says the same). */
export function reviewPercent(confirmed: number, total: number): number {
  if (!total) return 0;
  return confirmed >= total ? 100 : Math.min(99, Math.round(confirmed * 100 / total));
}
