import jmespath from "jmespath";
import Papa from "papaparse";

import type { FileRoute } from "./fileData";

const MAX_INPUT_BYTES = 64 * 1024 * 1024;
const MAX_ARRAY_RESULTS = 100;
const MAX_RESULT_BYTES = 256 * 1024;

type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export async function queryStructuredData(
  file: File,
  route: FileRoute,
  expression: string,
): Promise<{ value: JsonValue; truncated: boolean }> {
  if (file.size > MAX_INPUT_BYTES) {
    throw new Error(
      `Structured-data queries require parsing the complete file; ${file.size} bytes exceeds the ${MAX_INPUT_BYTES}-byte browser limit.`,
    );
  }

  const document = route === "csv" ? await csvToJson(file) : parseJson(await file.text());
  const result = jmespath.search(document, expression) as JsonValue | undefined;
  const value = result === undefined ? null : result;
  const bounded = Array.isArray(value) ? value.slice(0, MAX_ARRAY_RESULTS) : value;
  const truncated = Array.isArray(value) && value.length > MAX_ARRAY_RESULTS;
  if (new TextEncoder().encode(JSON.stringify(bounded)).byteLength > MAX_RESULT_BYTES) {
    throw new Error(`JMESPath result exceeds the ${MAX_RESULT_BYTES}-byte browser limit`);
  }
  return { value: bounded, truncated };
}

function parseJson(text: string): JsonValue {
  return JSON.parse(text) as JsonValue;
}

function csvToJson(file: File): Promise<JsonValue[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, JsonValue>>(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: "greedy",
      complete: (result) => {
        const fatal = result.errors.find(
          (error) => error.type === "Delimiter" || error.type === "Quotes",
        );
        if (fatal) reject(new Error(`CSV parse error: ${fatal.message}`));
        else if (!result.meta.fields?.length) reject(new Error("CSV has no header row"));
        else if (new Set(result.meta.fields).size !== result.meta.fields.length) {
          reject(new Error("CSV headers must be unique"));
        } else resolve(result.data);
      },
      error: (error) => reject(error),
    });
  });
}
