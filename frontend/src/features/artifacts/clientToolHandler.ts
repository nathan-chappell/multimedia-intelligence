import type { ConversationWorkspaceValue } from "./fileData";
import { authenticatedFetch } from "../../lib/config";

export interface ChatKitClientToolCall {
  name: string;
  params: Record<string, unknown>;
}

export async function executeFileClientTool(
  workspace: ConversationWorkspaceValue,
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
  workspace: ConversationWorkspaceValue,
  { name, params }: ChatKitClientToolCall,
): Promise<Record<string, unknown>> {
  await workspace.waitUntilReady();
  if (name === "list_workspace_files") {
    const page = integer(params, "page", 1, 1);
    const durableFiles = recordArray(params, "durableFiles");
    const localDurableIds = new Set(
      workspace.getFiles().flatMap((entry) =>
        entry.durableAssetId ? [entry.durableAssetId] : [],
      ),
    );
    const files = [
      ...workspace.getFiles().map((entry) => ({
        fileId: entry.durableAssetId ?? entry.id,
        name: entry.filename,
        mediaType: entry.mediaType,
        sizeBytes: entry.sizeBytes,
        route: entry.route,
        durability: entry.durability,
      })),
      ...durableFiles
        .filter(
          (entry) =>
            typeof entry.fileId === "string" && !localDurableIds.has(entry.fileId),
        )
        .map((entry) => ({
          fileId: entry.fileId,
          name: entry.filename,
          mediaType: entry.mediaType,
          sizeBytes: entry.sizeBytes,
          route: entry.route,
          durability: "included",
          reference: entry.reference,
          previewPath: entry.previewPath,
        })),
    ];
    const pageSize = 20;
    const start = (page - 1) * pageSize;
    return {
      page,
      pageSize,
      total: files.length,
      hasMore: start + pageSize < files.length,
      files: files.slice(start, start + pageSize),
    };
  }

  const fileId = fileIdParam(params);
  const entry = await workspace.resolveFile(fileId);
  if (!entry) {
    throw new Error(`File ${fileId} is unavailable`);
  }

  switch (name) {
    case "view_file": {
      const start = optionalNumber(params, "start", 0) ?? 0;
      const count = optionalNumber(params, "count", Number.MIN_VALUE);
      if (entry.route === "audio" || entry.route === "video") {
        const transcript = params.transcript;
        if (typeof transcript !== "object" || transcript === null) {
          throw new Error("The media transcript is unavailable");
        }
        return { fileId, route: entry.route, mode: "transcript", transcript };
      }
      if (entry.route === "image") {
      if (entry.durableAssetId && !entry.includeId) {
        await workspace.saveFile(entry.id);
      }
      const saved = entry.durableAssetId
        ? { asset_id: entry.durableAssetId, include_id: entry.includeId ?? null }
        : await saveBrowserFile(
            entry.file,
            entry.file.name,
            "Could not save workspace image",
          );
      return {
        fileId,
        route: entry.route,
        mode: "image",
        file: {
          fileId: saved.asset_id,
          filename: entry.file.name,
          mediaType: entry.file.type,
          sizeBytes: entry.file.size,
          durability: "included",
        },
      };
      }
      if (entry.route === "pdf") {
        if (params.start == null && params.count == null) {
          const saved = entry.durableAssetId
            ? { asset_id: entry.durableAssetId }
            : await saveBrowserFile(
                entry.file,
                entry.file.name,
                "Could not save workspace PDF",
              );
          return {
            fileId,
            route: entry.route,
            mode: "pdf",
            file: {
              fileId: saved.asset_id,
              filename: entry.file.name,
              mediaType: "application/pdf",
              sizeBytes: entry.file.size,
              durability: "included",
            },
          };
        }
        const startPage = Math.max(1, Math.trunc(start || 1));
        const endPage = count === undefined
          ? startPage
          : startPage + Math.max(1, Math.trunc(count)) - 1;
        const { extractPdfPageRange } = await import("./pdfTools");
        const pdf = await extractPdfPageRange(entry.file, { startPage, endPage });
        const stem = entry.file.name.replace(/\.pdf$/i, "").slice(0, 800);
        const filename = `${stem}-pages-${startPage}-${endPage}.pdf`;
        const saved = await saveBrowserFile(pdf, filename, "Could not prepare PDF pages", fileId);
        await workspace.refreshThreadFiles();
        return {
          fileId,
          route: entry.route,
          mode: "pdf",
          startPage,
          endPage,
          file: {
            fileId: saved.asset_id,
            filename,
            mediaType: "application/pdf",
            sizeBytes: pdf.size,
            durability: "included",
          },
        };
      }
      if (!["text", "markup", "json", "csv", "tabular"].includes(entry.route)) {
        throw new Error(`Viewing ${entry.route} files is unsupported`);
      }
      const { readTextChars } = await import("./textTools");
      const charStart = Math.trunc(start);
      const charCount = Math.trunc(count ?? 16_384);
      return {
        fileId,
        route: entry.route,
        mode: "text",
        start: charStart,
        count: charCount,
        text: await readTextChars(entry.file, charStart, charCount),
      };
    }
    case "query_data": {
      if (!["json", "csv", "tabular"].includes(entry.route)) {
        throw new Error("JMESPath queries require a JSON or CSV file");
      }
      const expression = requiredString(params, "jmespathExpression");
      const { queryStructuredData } = await import("./structuredDataTools");
      const queried = await queryStructuredData(entry.file, entry.route, expression);
      let savedFile: SavedAsset | undefined;
      if (params.saveOutput === true) {
        const output = new Blob([JSON.stringify(queried.value, null, 2)], {
          type: "application/json",
        });
        const stem = entry.file.name.replace(/\.[^.]+$/, "").slice(0, 800);
        savedFile = await saveBrowserFile(
          output,
          `${stem}-query-result.json`,
          "Could not save query output",
          fileId,
        );
        await workspace.refreshThreadFiles();
      }
      return {
        fileId,
        jmespathExpression: expression,
        ...queried,
        ...(savedFile ? { savedFileId: savedFile.asset_id } : {}),
      };
    }
    default:
      throw new Error(`Unsupported client tool: ${name}`);
  }
}

function requiredString(params: Record<string, unknown>, key: string): string {
  const value = params[key];
  if (typeof value !== "string" || !value.trim()) throw new Error(`${key} must be a string`);
  return value;
}

function fileIdParam(params: Record<string, unknown>): string {
  const value = params.fileId ?? params.workspaceFileId ?? params.assetId;
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("fileId must be a string");
  }
  return value;
}

function optionalNumber(
  params: Record<string, unknown>,
  key: string,
  minimum: number,
): number | undefined {
  const value = params[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(`${key} must be a number of at least ${minimum}`);
  }
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

interface SavedAsset {
  asset_id: string;
  include_id: string | null;
}

async function saveBrowserFile(
  blob: Blob,
  filename: string,
  fallback: string,
  sourceFileId?: string,
): Promise<SavedAsset> {
  const query = new URLSearchParams({ filename });
  if (sourceFileId) query.set("source_file_id", sourceFileId);
  const response = await authenticatedFetch(`/api/assets?${query}`, {
    method: "POST",
    headers: { "Content-Type": blob.type || "application/octet-stream" },
    body: blob,
  });
  if (!response.ok) throw new Error(await responseError(response, fallback));
  const saved = (await response.json()) as SavedAsset;
  if (!saved.asset_id || !saved.include_id) {
    throw new Error("Saved browser file was not attached to the conversation");
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
