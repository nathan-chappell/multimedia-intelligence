const DEFAULT_MAX_CHARS = 64 * 1024;

export async function readTextChars(
  file: Blob,
  start: number,
  count: number,
  maxChars = DEFAULT_MAX_CHARS,
): Promise<string> {
  if (!Number.isSafeInteger(start) || start < 0) {
    throw new Error("start must be a non-negative integer");
  }
  if (!Number.isSafeInteger(count) || count < 1 || count > maxChars) {
    throw new Error(`count must be between 1 and ${maxChars}`);
  }
  const reader = file.stream().pipeThrough(new TextDecoderStream()).getReader();
  let position = 0;
  let result = "";
  try {
    while (result.length < count) {
      const chunk = await reader.read();
      if (chunk.done) break;
      const chunkEnd = position + chunk.value.length;
      if (chunkEnd > start) {
        const localStart = Math.max(0, start - position);
        result += chunk.value.slice(localStart, localStart + count - result.length);
      }
      position = chunkEnd;
    }
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  return result;
}
