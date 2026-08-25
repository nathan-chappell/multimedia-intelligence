import type { FileWorkspaceValue } from "./fileWorkspace";
import { authenticatedFetch } from "../../lib/config";

export interface ChatKitClientToolCall {
  name: string;
  params: Record<string, unknown>;
}

export async function executeFileClientTool(
  workspace: FileWorkspaceValue,
  toolCall: ChatKitClientToolCall,
): Promise<Record<string, unknown>> {
  try {
    return { ok: true, ...(await execute(workspace, toolCall)) };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Unknown browser tool failure",
      tool: toolCall.name,
    };
  }
}

async function execute(
  workspace: FileWorkspaceValue,
  { name, params }: ChatKitClientToolCall,
): Promise<Record<string, unknown>> {
  if (name === "list_files") {
    const page = integer(params, "page", 1, 1);
    const durableFiles = recordArray(params, "durableFiles");
    const localDurableIds = new Set(
      workspace.files.flatMap((entry) =>
        entry.durableAssetId ? [entry.durableAssetId] : [],
      ),
    );
    const files = [
      ...workspace.files.map((entry) => ({
        assetId: entry.id,
        name: entry.file.name,
        mediaType: entry.file.type || "application/octet-stream",
        sizeBytes: entry.file.size,
        route: entry.route,
        durability: entry.durability,
        durableAssetId: entry.durableAssetId,
      })),
      ...durableFiles
        .filter(
          (entry) =>
            typeof entry.assetId === "string" && !localDurableIds.has(entry.assetId),
        )
        .map((entry) => ({
          assetId: entry.assetId,
          name: entry.filename,
          mediaType: entry.mediaType,
          sizeBytes: entry.sizeBytes,
          route: entry.route,
          durability: "included",
          durableAssetId: entry.assetId,
          reference: entry.reference,
          previewPath: entry.previewPath,
        })),
    ];
    const pageSize = 10;
    const start = (page - 1) * pageSize;
    return {
      page,
      pageSize,
      total: files.length,
      hasMore: start + pageSize < files.length,
      files: files.slice(start, start + pageSize),
    };
  }

  const assetId = requiredString(params, "assetId");
  const entry = workspace.getFile(assetId);
  if (!entry) throw new Error(`No staged file is registered for asset ID ${assetId}`);

  switch (name) {
    case "read_text_chars":
    case "json_chars": {
      const start = integer(params, "start", 0, 0);
      const { readTextChars } = await import("./textTools");
      return {
        assetId,
        start,
        text: await readTextChars(
          entry.file,
          start,
          integer(params, "count", 16_384, 1),
        ),
      };
    }
    case "query_structured_data": {
      if (entry.route !== "json" && entry.route !== "csv") {
        throw new Error("JMESPath queries require a JSON or CSV file");
      }
      const expression = requiredString(params, "expression");
      const { queryStructuredData } = await import("./structuredDataTools");
      return {
        assetId,
        expression,
        ...(await queryStructuredData(entry.file, entry.route, expression)),
      };
    }
    case "pdf_random_sample": {
      const startPage = integer(params, "startPage", 1, 1);
      const endPage = optionalInteger(params, "endPage", 1);
      const count = boundedInteger(params, "count", 5, 1, 10);
      const outputMode = oneOf(params, "outputMode", ["text_content", "as_files"] as const);
      const range = { startPage, endPage };
      if (outputMode === "text_content") {
        const { samplePdfText } = await import("./pdfTools");
        const sample = await samplePdfText(entry.file, range, count);
        return { assetId, mode: outputMode, ...sample };
      }

      if (!workspace.activeThreadId) {
        throw new Error("A conversation must be active before sampled pages can be saved");
      }
      const { samplePdfAsFile } = await import("./pdfTools");
      const sample = await samplePdfAsFile(entry.file, range, count);
      const stem = entry.file.name.replace(/\.pdf$/i, "").slice(0, 800);
      const filename = `${stem}-sample-${sample.sampledPages.join("-")}.pdf`;
      const saved = await saveSampledPdf(sample.file, filename, workspace.activeThreadId);
      workspace.registerArtifact(
        assetId,
        "pdf_part",
        `${entry.file.name} · sampled pages ${sample.sampledPages.join(", ")}`,
        sample.file,
      );
      return {
        assetId,
        mode: outputMode,
        pageCount: sample.pageCount,
        range: sample.range,
        sampledPages: sample.sampledPages,
        files: [
          {
            assetId: saved.asset_id,
            filename,
            mediaType: "application/pdf",
            sizeBytes: sample.file.size,
            durability: "included",
            originalPages: sample.sampledPages,
          },
        ],
      };
    }
    case "pdf_render_page": {
      const page = integer(params, "page", undefined, 1);
      const { renderPdfPage } = await import("./pdfTools");
      const image = await renderPdfPage(entry.file, page, number(params, "scale", 1.75));
      const artifact = workspace.registerArtifact(
        assetId,
        "pdf_page_image",
        `${entry.file.name} · page ${page}`,
        image,
      );
      return transientArtifactResult(artifact, image);
    }
    case "pdf_extract_range": {
      const startPage = integer(params, "startPage", undefined, 1);
      const endPage = integer(params, "endPage", undefined, startPage);
      const { extractPdfPageRange } = await import("./pdfTools");
      const pdf = await extractPdfPageRange(entry.file, { startPage, endPage });
      const artifact = workspace.registerArtifact(
        assetId,
        "pdf_part",
        `${entry.file.name} · pages ${startPage}-${endPage}`,
        pdf,
      );
      return transientArtifactResult(artifact, pdf);
    }
    default:
      throw new Error(`Unsupported client tool: ${name}`);
  }
}

function transientArtifactResult(
  artifact: ReturnType<FileWorkspaceValue["registerArtifact"]>,
  blob: Blob,
): Record<string, unknown> {
  return {
    artifactId: artifact.id,
    sourceAssetId: artifact.sourceAssetId,
    kind: artifact.kind,
    mediaType: blob.type,
    sizeBytes: blob.size,
    durability: "local_preview",
  };
}

function requiredString(params: Record<string, unknown>, key: string): string {
  const value = params[key];
  if (typeof value !== "string" || !value.trim()) throw new Error(`${key} must be a string`);
  return value;
}

function integer(
  params: Record<string, unknown>,
  key: string,
  fallback: number | undefined,
  minimum: number,
): number {
  const value = params[key] ?? fallback;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${key} must be an integer of at least ${minimum}`);
  }
  return value;
}

function optionalInteger(
  params: Record<string, unknown>,
  key: string,
  minimum: number,
): number | undefined {
  if (params[key] === undefined || params[key] === null) return undefined;
  return integer(params, key, undefined, minimum);
}

function boundedInteger(
  params: Record<string, unknown>,
  key: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const value = integer(params, key, fallback, minimum);
  if (value > maximum) throw new Error(`${key} must be at most ${maximum}`);
  return value;
}

function number(params: Record<string, unknown>, key: string, fallback: number): number {
  const value = params[key] ?? fallback;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${key} must be a number`);
  }
  return value;
}

function recordArray(
  params: Record<string, unknown>,
  key: string,
): Record<string, unknown>[] {
  const value = params[key];
  if (value === undefined) return [];
  if (
    !Array.isArray(value) ||
    !value.every((entry) => typeof entry === "object" && entry !== null && !Array.isArray(entry))
  ) {
    throw new Error(`${key} must be an object array`);
  }
  return value as Record<string, unknown>[];
}

function oneOf<const T extends readonly string[]>(
  params: Record<string, unknown>,
  key: string,
  choices: T,
): T[number] {
  const value = params[key];
  if (typeof value !== "string" || !choices.includes(value)) {
    throw new Error(`${key} must be one of ${choices.join(", ")}`);
  }
  return value;
}

interface SavedSample {
  asset_id: string;
  include_id: string | null;
}

async function saveSampledPdf(
  pdf: Blob,
  filename: string,
  threadId: string,
): Promise<SavedSample> {
  const query = new URLSearchParams({ filename, thread_id: threadId });
  const response = await authenticatedFetch(`/api/assets?${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/pdf" },
    body: pdf,
  });
  if (!response.ok) throw new Error(await responseError(response, "Could not save PDF sample"));
  const saved = (await response.json()) as SavedSample;
  if (!saved.asset_id || !saved.include_id) {
    throw new Error("Saved PDF sample was not attached to the conversation");
  }
  return saved;
}

async function responseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
}
