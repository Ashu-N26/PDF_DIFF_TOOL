const fs = require("fs");
const { PDFDocument, rgb, StandardFonts } = require("pdf-lib");

async function buildMergedOutput({ oldPdfPath, newPdfPath, diff, outputPath }) {
  const newBytes = fs.readFileSync(newPdfPath);
  const oldBytes = fs.readFileSync(oldPdfPath);

  // Base doc = NEW PDF (we annotate this)
  const baseDoc = await PDFDocument.load(newBytes);
  const oldDoc = await PDFDocument.load(oldBytes);

  // Create summary page (first)
  const merged = await PDFDocument.create();
  const font = await merged.embedFont(StandardFonts.Helvetica);

  // Summary Page
  const page = merged.addPage([595, 842]); // A4 portrait
  page.drawRectangle({ x: 20, y: 20, width: 555, height: 802, borderColor: rgb(0, 0, 0), borderWidth: 1 });
  page.drawText("PDF Change Summary", { x: 40, y: 790, size: 18, font });

  const s = diff.summary.counts;
  page.drawText(`Added: ${s.added}`, { x: 40, y: 760, size: 12, font, color: rgb(0, 0.6, 0) });
  page.drawText(`Removed: ${s.removed}`, { x: 140, y: 760, size: 12, font, color: rgb(0.8, 0, 0) });
  page.drawText(`Changed: ${s.changed}`, { x: 260, y: 760, size: 12, font, color: rgb(0.8, 0, 0) });

  let y = 730;
  const line = (t, color = rgb(0,0,0)) => {
    page.drawText(t, { x: 40, y, size: 10, font, color });
    y -= 14;
  };

  line("Legend:  RED = changed/removed   GREEN = added");

  // Show top N removed/changed entries (space-limited)
  for (const p of diff.pages) {
    const entries = p.changes.slice(0, 6); // small digest per page
    if (entries.length === 0) continue;
    if (y < 80) break;
    line(`Page ${p.pageIndex + 1}:`, rgb(0,0,0));
    entries.forEach(ch => {
      if (y < 60) return;
      if (ch.type === "changed") {
        line(`• CHANGED: "${truncate(ch.oldText)}" → "${truncate(ch.newText)}"`, rgb(0.8,0,0));
      } else if (ch.type === "removed") {
        line(`• REMOVED: "${truncate(ch.oldText)}"`, rgb(0.8,0,0));
      } else if (ch.type === "added") {
        line(`• ADDED:   "${truncate(ch.newText)}"`, rgb(0,0.6,0));
      }
    });
  }

  // Append annotated new pages
  const copied = await merged.copyPages(baseDoc, baseDoc.getPageIndices());
  copied.forEach((cp, idx) => {
    merged.addPage(cp);
    // Overlay annotations for that page
    const pageDiff = diff.pages[idx];
    if (!pageDiff) return;

    pageDiff.changes.forEach(ch => {
      const c = ch.type === "added" ? rgb(0, 0.6, 0) : rgb(0.9, 0, 0);
      cp.drawRectangle({
        x: ch.box.x,
        y: ch.box.y,
        width: ch.box.w,
        height: ch.box.h,
        borderColor: c,
        borderWidth: 1.5,
        color: undefined // no fill
      });
    });

    // (Optional) footer tag
    cp.drawText("Annotated by PDF Diff Tool", { x: 20, y: 12, size: 8, font, color: rgb(0.3, 0.3, 0.3) });
  });

  const bytes = await merged.save();
  fs.writeFileSync(outputPath, bytes);
}

function truncate(s, n = 70) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

module.exports = { buildMergedOutput };
