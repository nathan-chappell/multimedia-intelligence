import type { FileWorkspaceValue } from "./fileWorkspace";

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
  if (name === "list_included_files") {
    return {
      files: workspace.files.map((entry) => ({
        assetId: entry.id,
        name: entry.file.name,
        mediaType: entry.file.type || "application/octet-stream",
        sizeBytes: entry.file.size,
        route: entry.route,
        durability: "local_browser_only",
      })),
      warning:
        "These are staged browser files, not finalized bucket assets. Do not claim they are durable or provider-ready.",
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
    case "json_path": {
      const { queryJsonPath } = await import("./jsonTools");
      return {
        assetId,
        results: await queryJsonPath(entry.file, stringArray(params, "queries")),
      };
    }
    case "csv_head": {
      const { csvHead } = await import("./csvTools");
      return {
        assetId,
        head: await csvHead(entry.file, integer(params, "count", 10, 1)),
      };
    }
    case "csv_stats": {
      const { csvStats } = await import("./csvTools");
      return {
        assetId,
        stats: await csvStats(entry.file, optionalStringArray(params, "columns")),
      };
    }
    case "pdf_inspect": {
      const { inspectPdf } = await import("./pdfTools");
      return {
        assetId,
        inspection: await inspectPdf(entry.file, integer(params, "sampleCount", 8, 1)),
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
    durability: "transient_browser_only",
    nextStep:
      "Upload and finalize this derivative through the backend before using it as an OpenAI file input.",
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

function number(params: Record<string, unknown>, key: string, fallback: number): number {
  const value = params[key] ?? fallback;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${key} must be a number`);
  }
  return value;
}

function stringArray(params: Record<string, unknown>, key: string): string[] {
  const value = params[key];
  if (
    !Array.isArray(value) ||
    !value.length ||
    !value.every((entry) => typeof entry === "string")
  ) {
    throw new Error(`${key} must be a non-empty string array`);
  }
  return value;
}

function optionalStringArray(params: Record<string, unknown>, key: string): string[] {
  const value = params[key];
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string")) {
    throw new Error(`${key} must be a string array`);
  }
  return value;
}
