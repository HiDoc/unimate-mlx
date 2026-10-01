import assert from "node:assert/strict";
import test from "node:test";
import { formatElapsedTime, movePrompt, parsePromptQueue } from "./studio-utils.ts";

test("parsePromptQueue trims prompts, drops blank lines, and labels in order", () => {
  assert.deepEqual(parsePromptQueue("  Walk forward  \n\n Turn left\r\n  "), [
    { text: "Walk forward", label: "Prompt 1" },
    { text: "Turn left", label: "Prompt 2" },
  ]);
});

test("parsePromptQueue returns an empty queue for empty input", () => {
  assert.deepEqual(parsePromptQueue(" \n\r\n "), []);
});

test("movePrompt reorders an item up or down without changing the source queue", () => {
  const queue = [
    { text: "Walk", label: "A" },
    { text: "Turn", label: "B" },
    { text: "Jump", label: "C" },
  ];

  assert.deepEqual(movePrompt(queue, 1, -1), [queue[1], queue[0], queue[2]]);
  assert.deepEqual(movePrompt(queue, 1, 1), [queue[0], queue[2], queue[1]]);
  assert.deepEqual(queue, [
    { text: "Walk", label: "A" },
    { text: "Turn", label: "B" },
    { text: "Jump", label: "C" },
  ]);
});

test("movePrompt preserves order at the ends and for invalid indices", () => {
  const queue = [{ text: "Only", label: "A" }];
  assert.deepEqual(movePrompt(queue, 0, -1), queue);
  assert.deepEqual(movePrompt(queue, 0, 1), queue);
  assert.deepEqual(movePrompt(queue, -1, 1), queue);
  assert.deepEqual(movePrompt(queue, 1, -1), queue);
});

test("formatElapsedTime displays minutes and seconds, then hours when needed", () => {
  assert.equal(formatElapsedTime(0), "00:00");
  assert.equal(formatElapsedTime(9), "00:09");
  assert.equal(formatElapsedTime(65), "01:05");
  assert.equal(formatElapsedTime(3599), "59:59");
  assert.equal(formatElapsedTime(3600), "1:00:00");
  assert.equal(formatElapsedTime(3661), "1:01:01");
});

test("formatElapsedTime clamps negative and non-finite values to zero", () => {
  assert.equal(formatElapsedTime(-3), "00:00");
  assert.equal(formatElapsedTime(Number.NaN), "00:00");
  assert.equal(formatElapsedTime(Number.POSITIVE_INFINITY), "00:00");
});
