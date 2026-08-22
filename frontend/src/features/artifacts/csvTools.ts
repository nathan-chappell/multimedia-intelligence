import Papa from "papaparse";

const MAX_CSV_BYTES = 64 * 1024 * 1024;
const MAX_HEAD_ROWS = 20;
const MAX_SAMPLE_VALUES = 10_000;

type CsvRow = Record<string, string>;
type InferredType =
  | "integer"
  | "number"
  | "boolean"
  | "datetime"
  | "string"
  | "unknown";

export interface CsvColumnSummary {
  name: string;
  inferredType: InferredType;
  nullable: boolean;
}

export interface CsvHeadResult {
  columns: CsvColumnSummary[];
  rows: Record<string, string | number | boolean | null>[];
  sampledRowCount: number;
}

export interface CsvNumericStats {
  column: string;
  count: number;
  nullCount: number;
  invalidCount: number;
  minimum: number;
  maximum: number;
  mean: number;
  standardDeviation: number | null;
  quantiles: { p25: number; p50: number; p75: number };
  approximateQuantiles: boolean;
}

export async function csvHead(file: File, count = 10): Promise<CsvHeadResult> {
  assertCsvSize(file);
  if (!Number.isSafeInteger(count) || count < 1 || count > MAX_HEAD_ROWS) {
    throw new Error(`count must be between 1 and ${MAX_HEAD_ROWS}`);
  }
  const rows = await parseCsv(file, 200);
  const headers = Object.keys(rows[0] ?? {});
  if (headers.length === 0) throw new Error("CSV has no header row");
  const columns = headers.map((name) => inferColumn(name, rows));
  const typeByName = new Map(columns.map((column) => [column.name, column.inferredType]));
  return {
    columns,
    rows: rows.slice(0, count).map((row) =>
      Object.fromEntries(
        headers.map((header) => [header, coerceValue(row[header] ?? "", typeByName.get(header))]),
      ),
    ),
    sampledRowCount: rows.length,
  };
}

export async function csvStats(
  file: File,
  requestedColumns: readonly string[] = [],
): Promise<CsvNumericStats[]> {
  assertCsvSize(file);
  const rows = await parseCsv(file);
  const headers = Object.keys(rows[0] ?? {});
  if (headers.length === 0) throw new Error("CSV has no header row");
  const requested = requestedColumns.length
    ? [...requestedColumns]
    : headers.filter((header) => {
        const inferred = inferColumn(header, rows.slice(0, 200));
        return inferred.inferredType === "integer" || inferred.inferredType === "number";
      });
  const unknown = requested.filter((column) => !headers.includes(column));
  if (unknown.length) throw new Error(`Unknown CSV columns: ${unknown.join(", ")}`);
  if (!requested.length) throw new Error("No numeric columns were inferred");
  return requested.map((column) => summarizeColumn(column, rows));
}

function parseCsv(file: File, preview?: number): Promise<CsvRow[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<CsvRow>(file, {
      header: true,
      skipEmptyLines: "greedy",
      preview,
      complete: (result) => {
        const fatal = result.errors.find((error) => error.type === "Delimiter" || error.type === "Quotes");
        if (fatal) reject(new Error(`CSV parse error: ${fatal.message}`));
        else resolve(result.data);
      },
      error: (error) => reject(error),
    });
  });
}

function inferColumn(name: string, rows: readonly CsvRow[]): CsvColumnSummary {
  const values = rows.map((row) => (row[name] ?? "").trim());
  const present = values.filter((value) => value !== "" && value.toLocaleLowerCase() !== "null");
  let inferredType: InferredType = "unknown";
  if (present.length) {
    if (present.every((value) => /^[-+]?\d+$/.test(value))) inferredType = "integer";
    else if (present.every((value) => Number.isFinite(Number(value)))) inferredType = "number";
    else if (present.every((value) => /^(true|false|yes|no)$/i.test(value))) inferredType = "boolean";
    else if (present.every(isIsoDateTime)) inferredType = "datetime";
    else inferredType = "string";
  }
  return { name, inferredType, nullable: present.length !== values.length };
}

function isIsoDateTime(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-]+(?:Z)?)?$/.test(value)) return false;
  return Number.isFinite(Date.parse(value));
}

function coerceValue(value: string, inferredType?: InferredType): string | number | boolean | null {
  const normalized = value.trim();
  if (normalized === "" || normalized.toLocaleLowerCase() === "null") return null;
  if (inferredType === "integer" || inferredType === "number") return Number(normalized);
  if (inferredType === "boolean") return /^(true|yes)$/i.test(normalized);
  return value;
}

function summarizeColumn(column: string, rows: readonly CsvRow[]): CsvNumericStats {
  let count = 0;
  let nullCount = 0;
  let invalidCount = 0;
  let mean = 0;
  let m2 = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  const sample: number[] = [];
  rows.forEach((row) => {
    const raw = (row[column] ?? "").trim();
    if (raw === "" || raw.toLocaleLowerCase() === "null") {
      nullCount += 1;
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      invalidCount += 1;
      return;
    }
    count += 1;
    const delta = value - mean;
    mean += delta / count;
    m2 += delta * (value - mean);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (sample.length < MAX_SAMPLE_VALUES) sample.push(value);
    else {
      const replacement = Math.floor(Math.random() * count);
      if (replacement < MAX_SAMPLE_VALUES) sample[replacement] = value;
    }
  });
  if (!count) throw new Error(`Column ${column} has no finite numeric values`);
  sample.sort((left, right) => left - right);
  return {
    column,
    count,
    nullCount,
    invalidCount,
    minimum,
    maximum,
    mean,
    standardDeviation: count > 1 ? Math.sqrt(m2 / (count - 1)) : null,
    quantiles: {
      p25: quantile(sample, 0.25),
      p50: quantile(sample, 0.5),
      p75: quantile(sample, 0.75),
    },
    approximateQuantiles: count > MAX_SAMPLE_VALUES,
  };
}

function quantile(values: readonly number[], fraction: number): number {
  const position = (values.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return values[lower];
  return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

function assertCsvSize(file: File): void {
  if (file.size > MAX_CSV_BYTES) {
    throw new Error(`Browser CSV analysis is capped at ${MAX_CSV_BYTES} bytes`);
  }
}
