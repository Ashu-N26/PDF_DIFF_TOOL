const fs = require("fs");
const path = require("path");
const pdfjs = require("pdfjs-dist/legacy/build/pdf.js");

// Tell pdfjs to use its internal worker in Node
pdfjs.GlobalWorkerOptions.workerSrc =
  require("pdfjs-dist/legacy/build/pdf.worker.js");

async function extractDocument(pdfPath) {
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const loadingTask = pdfjs.getDocument({ data, useSystemFonts: true });
  const doc = await loadingTask.promise;

  const pages = [];
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const viewport = page.getViewport({ scale: 1.0 });
    const textContent = await page.getTextContent();

    // Collect words with boxes (merge chars into words by spacing)
    const words = [];
    let current = null;

    const pushCurrent = () => {
      if (current) {
        // Normalize y to PDF coordinate space (pdfjs gives top-left y)
        const yPdf = viewport.height - (current.y + current.h);
        words.push({ text: current.text, x: current.x, y: yPdf, w: current.w, h: current.h });
      }
      current = null;
    };

    for (const item of textContent.items) {
      const str = item.str;
      const tx = pdfjs.Util.transform(viewport.transform, item.transform);
      // tx = [a, b, c, d, e, f] → origin at (e, f), width ~ item.width, height ~ fontSize
      const x = tx[4], yTop = tx[5];
      const h = Math.abs(tx[3]);
      const w = item.width;

      const tokens = str.split(/(\s+)/).filter(t => t.length > 0);
      let cx = x;
      tokens.forEach(tok => {
        const isSpace = /^\s+$/.test(tok);
        const tokWidth = (w * tok.length) / str.length;

        if (isSpace) {
          pushCurrent();
        } else {
          if (!current) current = { text: tok, x: cx, y: yTop - h, w: tokWidth, h };
          else {
            // merge contiguous token into a word
            current.text += tok;
            current.w = (cx + tokWidth) - current.x;
          }
        }
        cx += tokWidth;
      });
      pushCurrent();
    }

    pages.push({
      width: viewport.width,
      height: viewport.height,
      words
    });
  }

  return { pageCount: pages.length, pages };
}

module.exports = { extractDocument };
