import { JSONPath } from "jsonpath-plus";

const DEFAULT_MAX_JSON_BYTES = 32 * 1024 * 1024;
const DEFAULT_MAX_RESULTS = 100;
const DEFAULT_MAX_RESULT_BYTES = 256 * 1024;

export interface JsonPathQueryResult {
  query: string;
  values: unknown[];
  truncated: boolean;
}

export async function queryJsonPath(
  file: Blob,
  queryOrQueries: string | readonly string[],
  options: {
    maxJsonBytes?: number;
    maxResults?: number;
    maxResultBytes?: number;
  } = {},
): Promise<JsonPathQueryResult[]> {
  const maxJsonBytes = options.maxJsonBytes ?? DEFAULT_MAX_JSON_BYTES;
  const maxResults = options.maxResults ?? DEFAULT_MAX_RESULTS;
  const maxResultBytes = options.maxResultBytes ?? DEFAULT_MAX_RESULT_BYTES;
  if (file.size > maxJsonBytes) {
    throw new Error(
      `JSONPath requires parsing the complete document; ${file.size} bytes exceeds the ${maxJsonBytes}-byte browser limit.`,
    );
  }

  const queries = typeof queryOrQueries === "string" ? [queryOrQueries] : [...queryOrQueries];
  if (queries.length < 1 || queries.length > 8) {
    throw new Error("Provide between 1 and 8 JSONPath queries");
  }

  const document = JSON.parse(await file.text()) as object;
  const encoder = new TextEncoder();
  return queries.map((query) => {
    const allValues = JSONPath({ path: query, json: document, wrap: true }) as unknown[];
    const values: unknown[] = [];
    let encodedBytes = 2;
    for (const value of allValues) {
      if (values.length >= maxResults) break;
      const encoded = JSON.stringify(value);
      const valueBytes = encoder.encode(encoded).byteLength;
      if (encodedBytes + valueBytes > maxResultBytes) break;
      values.push(value);
      encodedBytes += valueBytes;
    }
    return {
      query,
      values,
      truncated: values.length !== allValues.length,
    };
  });
}
