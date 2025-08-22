const express = require("express");
const multer = require("multer");
const path = require("path");
const cors = require("cors");

const { extractWords } = require("./utils/pdfExtract");
const { computeChanges } = require("./utils/diffEngine");
const { renderMergedDiff } = require("./utils/pdfRender");

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(cors());
app.use(express.static(path.join(__dirname, "public")));

app.get("/health", (_, res) => res.json({ ok: true }));

app.post("/api/compare", upload.fields([{ name: "pdf1" }, { name: "pdf2" }]), async (req, res) => {
  try {
    if (!req.files?.pdf1?.[0] || !req.files?.pdf2?.[0]) {
      return res.status(400).json({ error: "Upload pdf1 and pdf2" });
    }

    const oldBuf = req.files.pdf1[0].buffer;
    const newBuf = req.files.pdf2[0].buffer;

    // Extract word tokens with bounding boxes
    const oldTokens = await extractWords(oldBuf);
    const newTokens = await extractWords(newBuf);

    // Compute adds/removes + numeric value pairs (old→new) for summary
    const changes = computeChanges(oldTokens, newTokens); // { added, removed, valuePairs, byPage }

    // Render: Summary page + diff rectangles on top of NEW pdf pages
    const outPdf = await renderMergedDiff(newBuf, changes);

    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", 'attachment; filename="diff.pdf"');
    res.send(Buffer.from(outPdf));
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: String(err?.message || err) });
  }
});

// Serve UI
app.get("/", (_, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`PDF Diff running on :${PORT}`));

