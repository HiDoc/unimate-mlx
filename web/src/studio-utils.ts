import type { PromptItem } from "./types";

/** Parse a textarea containing one prompt per line into queue items. */
export function parsePromptQueue(source: string): PromptItem[] {
  return source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((text, index) => ({ text, label: `Prompt ${index + 1}` }));
}

/** Move an item one place in either direction, preserving the original queue. */
export function movePrompt(
  queue: PromptItem[],
  index: number,
  direction: -1 | 1,
): PromptItem[] {
  const next = [...queue];
  const target = index + direction;
  if (index < 0 || index >= queue.length || target < 0 || target >= queue.length) {
    return next;
  }

  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

/** Format elapsed seconds as MM:SS, or H:MM:SS for durations over an hour. */
export function formatElapsedTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainder = safeSeconds % 60;
  const two = (value: number) => String(value).padStart(2, "0");

  return hours > 0
    ? `${hours}:${two(minutes)}:${two(remainder)}`
    : `${two(minutes)}:${two(remainder)}`;
}
