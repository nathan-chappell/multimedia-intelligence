import { PDFDocument } from "pdf-lib";
import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentProxy,
} from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

// Client-tool dispatch stays on the main thread. This URL is only for PDF.js's
// internal parser/rendering worker, which getDocument() creates and manages.
GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const DEFAULT_EXTRACTION_LIMIT = 256 * 1024 * 1024;

export interface PdfPageProbe {
  page: number;
  text: string;
  truncated: boolean;
}

export interface PdfTextSample {
  pageCount: number;
  range: Required<PdfPageRange>;
  pages: PdfPageProbe[];
}

export interface PdfFileSample {
  pageCount: number;
  range: Required<PdfPageRange>;
  sampledPages: number[];
  file: Blob;
}

export interface PdfPageRange {
  startPage: number;
  endPage?: number;
}

const MAX_TEXT_CHARACTERS_PER_PAGE = 16_384;

export async function samplePdfText(
  file: Blob,
  range: PdfPageRange,
  count: number,
): Promise<PdfTextSample> {
  return withPdfDocument(file, async (document) => {
    const resolvedRange = resolvePageRange(document.numPages, range);
    const sampledPageNumbers = randomSamplePages(resolvedRange, count);
    const pages = await Promise.all(
      sampledPageNumbers.map(async (pageNumber): Promise<PdfPageProbe> => {
        const page = await document.getPage(pageNumber);
        try {
          const textContent = await page.getTextContent();
          const text = textContent.items
            .map((item) => ("str" in item ? item.str : ""))
            .join(" ")
            .trim();
          return {
            page: pageNumber,
            text: text.slice(0, MAX_TEXT_CHARACTERS_PER_PAGE),
            truncated: text.length > MAX_TEXT_CHARACTERS_PER_PAGE,
          };
        } finally {
          page.cleanup();
        }
      }),
    );
    return {
      pageCount: document.numPages,
      range: resolvedRange,
      pages,
    };
  });
}

export async function samplePdfAsFile(
  file: Blob,
  range: PdfPageRange,
  count: number,
  maxSourceBytes = DEFAULT_EXTRACTION_LIMIT,
): Promise<PdfFileSample> {
  if (file.size > maxSourceBytes) {
    throw new Error(
      `Page extraction loads the PDF in browser memory; ${file.size} bytes exceeds the ${maxSourceBytes}-byte limit.`,
    );
  }
  const source = await PDFDocument.load(await file.arrayBuffer(), {
    ignoreEncryption: false,
    updateMetadata: false,
  });
  const pageCount = source.getPageCount();
  const resolvedRange = resolvePageRange(pageCount, range);
  const sampledPages = randomSamplePages(resolvedRange, count);
  const output = await PDFDocument.create();
  const pages = await output.copyPages(
    source,
    sampledPages.map((page) => page - 1),
  );
  pages.forEach((page) => output.addPage(page));
  const saved = await output.save();
  const ownedBuffer = new ArrayBuffer(saved.byteLength);
  new Uint8Array(ownedBuffer).set(saved);
  return {
    pageCount,
    range: resolvedRange,
    sampledPages,
    file: new Blob([ownedBuffer], { type: "application/pdf" }),
  };
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
  const resolvedRange = resolvePageRange(source.getPageCount(), range);
  const output = await PDFDocument.create();
  const indices = Array.from(
    { length: resolvedRange.endPage - resolvedRange.startPage + 1 },
    (_unused, index) => resolvedRange.startPage - 1 + index,
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

function randomSamplePages(range: Required<PdfPageRange>, requested: number): number[] {
  if (!Number.isSafeInteger(requested) || requested < 1 || requested > 10) {
    throw new Error("count must be an integer between 1 and 10");
  }
  const available = range.endPage - range.startPage + 1;
  const count = Math.min(requested, available);
  const selected = new Set<number>();
  for (let page = range.endPage - count + 1; page <= range.endPage; page += 1) {
    const candidate = randomInteger(range.startPage, page);
    selected.add(selected.has(candidate) ? page : candidate);
  }
  return [...selected].sort((left, right) => left - right);
}

function resolvePageRange(
  pageCount: number,
  range: PdfPageRange,
): Required<PdfPageRange> {
  const endPage = range.endPage ?? pageCount;
  if (
    !Number.isSafeInteger(range.startPage) ||
    !Number.isSafeInteger(endPage) ||
    range.startPage < 1 ||
    endPage < range.startPage ||
    endPage > pageCount
  ) {
    throw new Error(`Page range must be within 1-${pageCount}`);
  }
  return { startPage: range.startPage, endPage };
}

function validatePageRange(pageCount: number, range: PdfPageRange): void {
  resolvePageRange(pageCount, range);
}

function randomInteger(minimum: number, maximum: number): number {
  const size = maximum - minimum + 1;
  const limit = Math.floor(0x1_0000_0000 / size) * size;
  const values = new Uint32Array(1);
  do crypto.getRandomValues(values); while (values[0] >= limit);
  return minimum + (values[0] % size);
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
