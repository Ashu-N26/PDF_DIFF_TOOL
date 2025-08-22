const { PDFDocument, rgb, StandardFonts } = require("pdf-lib");

// Draw rectangle with stroke color
function drawBox(page, t, color, thickness = 1) {
  const pad = 1.5;
  page.drawRectangle({
    x: t.x - pad,
    y: t.y - t.height - pad,
    width: t.width + pad * 2,
    height: t.height + pad * 2,
    borderWidth: thickness,
    borderColor: color,
    color: undefined,
    opacity: 1
  });
}

function toRgb(hex) {
  if (hex === "red") return rgb(0.95, 0.1, 0.1);
  if (hex === "green") return rgb(0.15, 0.7, 0.2);
  if (hex === "orange") return rgb(1, 0.5, 0);
  return rgb(0,0,0);
}

async function renderMergedDiff(newPdfBuffer, changes) {
  const doc = await PDFDocument.load(newPdfBuffer);
  const font = await doc.embedFont(StandardFonts.Helvetica);

  // Build a Summary page (can spill to more if long)
  const pageW = 595.28, pageH = 841.89;
  let summary = doc.insertPage(0, [pageW, pageH]);

  let y = pageH - 40;
  const line = (txt, size = 12, color = rgb(0,0,0)) => {
    summary.drawText(txt, { x: 40, y, size, color, font });
    y -= size + 6;
    if (y < 60) {
      summary = doc.addPage([pageW, pageH]);
      y = pageH - 40;
    }
  };

  line("PDF Comparison Summary", 18);
  line("Legend: Added = green, Removed = red, Value changed = red (old) → green (new)", 10, rgb(0.2,0.2,0.2));
  line("");

  // Page totals
  const pages = Object.keys(changes.byPage).map(n => +n).sort((a,b)=>a-b);
  for (const p of pages) {
    const pg = changes.byPage[p];
    const a = pg.added.length;
    const r = pg.removed.length;
    const v = pg.valuePairs.length;
    line(`Page ${p}: +${a}  -${r}  ↦${v}`);
    // Show first few value pairs
    for (const vp of pg.valuePairs.slice(0, 6)) {
      line(` • Value: "${vp.old.text}" → "${vp.new.text}" (near y=${Math.round(vp.new.y)})`, 10, rgb(0.25,0.25,0.25));
    }
  }

  // Now overlay rectangles on the real pages (the "new" pdf)
  const red = toRgb("red");
  const green = toRgb("green");

  for (const p of pages) {
    const page = doc.getPage(p); // after insert summary at 0, original page #n is now at index n (1-based caller) => pdf-lib uses 0-based; getPage(p) is correct because we inserted one page at 0, so new pdf page 1 is old page 0.
    const pg = changes.byPage[p];

    // removed (old only) -> draw red where it used to be (approximate position in new doc)
    for (const t of pg.removed) drawBox(page, t, red, 1.2);
    // added (new only) -> green
    for (const t of pg.added) drawBox(page, t, green, 1.2);
    // value pairs -> emphasize both old red and new green; also connector arrow
    for (const pair of pg.valuePairs) {
      drawBox(page, pair.old, red, 1.5);
      drawBox(page, pair.new, green, 1.5);
      // small arrow
      const midY = pair.new.y - pair.new.height / 2;
      page.drawLine({
        start: { x: pair.old.x + pair.old.width, y: midY },
        end: { x: pair.new.x, y: midY },
        thickness: 0.8,
        color: toRgb("orange")
      });
    }
  }

  return await doc.save();
}

module.exports = { renderMergedDiff };
