// Extract words with approximate bounding boxes using pdfjs-dist (Node)
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");

// Helper to split a text item into per-word boxes
function splitIntoWordBoxes(item, viewport) {
  const words = [];
  const str = item.str || "";
  if (!str.trim()) return words;

  const fullWidth = item.width || Math.abs(item.transform[0]) * str.length || 0;
  const charW = fullWidth / Math.max(str.length, 1);

  const segments = str.match(/\S+|\s+/g) || [];
  let cursorX = item.transform[4]; // e
  const pageHeight = viewport.height;

  for (const seg of segments) {
    const isSpace = /^\s+$/.test(seg);
    const w = isSpace ? charW * seg.length : charW * seg.length;
    if (!isSpace) {
      words.push({
        text: seg,
        x: cursorX,
        y: pageHeight - item.transform[5], // flip y
        width: w,
        height: Math.max(item.height || Math.abs(item.transform[3]) || 10, 8),
      });
    }
    cursorX += w;
  }
  return words;
}

async function extractWords(pdfBuffer) {
  const loadingTask = pdfjsLib.getDocument({ data: pdfBuffer, useSystemFonts: true });
  const doc = await loadingTask.promise;
  const tokens = []; // {page, text, x,y,w,h}

  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const viewport = page.getViewport({ scale: 1.0 });
    const content = await page.getTextContent({ normalizeWhitespace: true });
    for (const item of content.items) {
      const boxes = splitIntoWordBoxes(item, viewport);
      for (const b of boxes) {
        tokens.push({
          page: p,
          text: b.text,
          x: b.x,
          y: b.y,
          width: b.width,
          height: b.height
        });
      }
    }
    page.cleanup();
  }
  await doc.destroy();
  return tokens;
}

module.exports = { extractWords };

