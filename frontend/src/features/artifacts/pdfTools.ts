import { PDFDocument } from "pdf-lib";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
} from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const DEFAULT_EXTRACTION_LIMIT = 256 * 1024 * 1024;

export interface PdfPageProbe {
  page: number;
  textCharacters: number;
  textPreview: string;
}

export interface PdfInspection {
  pageCount: number;
  sampledPages: PdfPageProbe[];
  likelyTextPdf: boolean;
}

export interface PdfPageRange {
  startPage: number;
  endPage: number;
}

export async function inspectPdf(
  file: Blob,
  sampleCount = 8,
): Promise<PdfInspection> {
  return withPdfDocument(file, async (document) => {
    const sampledPageNumbers = selectSamplePages(document.numPages, sampleCount);
    const sampledPages = await Promise.all(
      sampledPageNumbers.map(async (pageNumber): Promise<PdfPageProbe> => {
        const page = await document.getPage(pageNumber);
        const textContent = await page.getTextContent();
        const text = textContent.items
          .map((item) => ("str" in item ? item.str : ""))
          .join(" ")
          .replace(/\s+/g, " ")
          .trim();
        page.cleanup();
        return {
          page: pageNumber,
          textCharacters: text.length,
          textPreview: text.slice(0, 500),
        };
      }),
    );
    const pagesWithText = sampledPages.filter((page) => page.textCharacters >= 80).length;
    return {
      pageCount: document.numPages,
      sampledPages,
      likelyTextPdf: pagesWithText >= Math.ceil(sampledPages.length * 0.6),
    };
  });
}

export async function renderPdfPage(
  file: Blob,
  pageNumber: number,
  scale = 1.75,
): Promise<Blob> {
  if (!Number.isFinite(scale) || scale < 0.5 || scale > 4) {
    throw new Error("scale must be between 0.5 and 4");
  }
  return withPdfDocument(file, async (document) => {
    validatePageRange(document.numPages, { startPage: pageNumber, endPage: pageNumber });
    const page = await document.getPage(pageNumber);
    const viewport = page.getViewport({ scale });
    const canvas = documentCanvas(viewport.width, viewport.height);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D rendering is unavailable");
    await page.render({ canvas, canvasContext: context, viewport }).promise;
    const result = await canvasToBlob(canvas, "image/png");
    page.cleanup();
    canvas.width = 0;
    canvas.height = 0;
    return result;
  });
}

export async function extractPdfPageRange(
  file: Blob,
  range: PdfPageRange,
  maxSourceBytes = DEFAULT_EXTRACTION_LIMIT,
): Promise<Blob> {
  if (file.size > maxSourceBytes) {
    throw new Error(
      `Page extraction loads the PDF in browser memory; ${file.size} bytes exceeds the ${maxSourceBytes}-byte limit.`,
    );
  }
  const source = await PDFDocument.load(await file.arrayBuffer(), {
    ignoreEncryption: false,
    updateMetadata: false,
  });
  validatePageRange(source.getPageCount(), range);
  const output = await PDFDocument.create();
  const indices = Array.from(
    { length: range.endPage - range.startPage + 1 },
    (_unused, index) => range.startPage - 1 + index,
  );
  const pages = await output.copyPages(source, indices);
  pages.forEach((page) => output.addPage(page));
  const saved = await output.save();
  const ownedBuffer = new ArrayBuffer(saved.byteLength);
  new Uint8Array(ownedBuffer).set(saved);
  return new Blob([ownedBuffer], { type: "application/pdf" });
}

async function withPdfDocument<T>(
  file: Blob,
  operation: (document: PDFDocumentProxy) => Promise<T>,
): Promise<T> {
  const url = URL.createObjectURL(file);
  const loadingTask = getDocument({ url });
  try {
    const document = await loadingTask.promise;
    return await operation(document);
  } finally {
    await loadingTask.destroy();
    URL.revokeObjectURL(url);
  }
}

function selectSamplePages(pageCount: number, requested: number): number[] {
  const count = Math.max(1, Math.min(requested, pageCount, 20));
  if (count === 1) return [1];
  return Array.from(
    new Set(
      Array.from({ length: count }, (_unused, index) =>
        Math.round(1 + (index * (pageCount - 1)) / (count - 1)),
      ),
    ),
  );
}

function validatePageRange(pageCount: number, range: PdfPageRange): void {
  if (
    !Number.isSafeInteger(range.startPage) ||
    !Number.isSafeInteger(range.endPage) ||
    range.startPage < 1 ||
    range.endPage < range.startPage ||
    range.endPage > pageCount
  ) {
    throw new Error(`Page range must be within 1-${pageCount}`);
  }
}

function documentCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(width);
  canvas.height = Math.ceil(height);
  return canvas;
}

function canvasToBlob(canvas: HTMLCanvasElement, mediaType: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("PDF page rendering produced no image"));
    }, mediaType);
  });
}
