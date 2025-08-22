const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs-extra");
const cors = require("cors");
const { v4: uuidv4 } = require("uuid");

const { extractDocument } = require("./utils/pdfExtract");
const { diffDocuments } = require("./utils/pdfDiff");
const { buildMergedOutput } = require("./utils/mergePdf");

const app = express();
const PORT = process.env.PORT || 3000;

// CORS + static viewer
app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true, limit: "20mb" }));

// Serve the viewer
app.use("/", express.static(path.join(__dirname, "public")));

// uploads/results dirs
const UPLOAD_DIR = path.join(__dirname, "uploads");
const RESULT_DIR = path.join(__dirname, "results");
fs.ensureDirSync(UPLOAD_DIR);
fs.ensureDirSync(RESULT_DIR);

// Multer config
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB per file
  fileFilter: (_req, file, cb) => {
    if (file.mimetype !== "application/pdf") return cb(new Error("PDFs only"));
    cb(null, true);
  }
});

// API: compare two PDFs → JSON + merged PDF on disk
app.post(
  "/api/compare",
  upload.fields([{ name: "pdf1", maxCount: 1 }, { name: "pdf2", maxCount: 1 }]),
  async (req, res) => {
    try {
      if (!req.files?.pdf1?.[0] || !req.files?.pdf2?.[0]) {
        return res.status(400).json({ error: "Both PDFs (pdf1 & pdf2) are required." });
      }

      const runId = uuidv4();
      const outDir = path.join(RESULT_DIR, runId);
      await fs.ensureDir(outDir);

      const oldPath = req.files.pdf1[0].path;
      const newPath = req.files.pdf2[0].path;

      // 1) Parse PDFs to text + boxes
      const oldDoc = await extractDocument(oldPath);
      const newDoc = await extractDocument(newPath);

      // 2) Diff at word-level with positions
      const result = diffDocuments(oldDoc, newDoc);

      // 3) Build merged PDF (summary + annotated pages)
      const mergedPdfPath = path.join(outDir, "merged.pdf");
      await buildMergedOutput({
        oldPdfPath: oldPath,
        newPdfPath: newPath,
        diff: result,
        outputPath: mergedPdfPath
      });

      // 4) Save JSON for the viewer
      const jsonPath = path.join(outDir, "diff.json");
      await fs.writeJson(jsonPath, { runId, ...result }, { spaces: 2 });

      res.json({
        runId,
        summary: result.summary,
        pages: result.pages,
        mergedPdfUrl: `/results/${runId}/merged.pdf`,
        diffJsonUrl: `/results/${runId}/diff.json`
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err?.message || "Compare failed" });
    }
  }
);

// Serve results static
app.use("/results", express.static(RESULT_DIR));

// Health
app.get("/healthz", (_req, res) => res.send("ok"));

app.listen(PORT, () => {
  console.log(`PDF Diff Tool listening on :${PORT}`);
});
