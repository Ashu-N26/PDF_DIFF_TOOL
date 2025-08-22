const express = require("express");
const path = require("path");
const multer = require("multer");
const fs = require("fs");
const pdfParse = require("pdf-parse");
const diff = require("diff");

const app = express();
const PORT = process.env.PORT || 10000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Multer setup for uploads
const upload = multer({ dest: "uploads/" });

// Serve frontend
app.use(express.static(path.join(__dirname)));

// Compare PDFs
app.post("/compare", upload.array("pdfs", 2), async (req, res) => {
  if (!req.files || req.files.length !== 2) {
    return res.status(400).json({ error: "Upload exactly 2 PDF files" });
  }

  try {
    const pdf1Buffer = fs.readFileSync(req.files[0].path);
    const pdf2Buffer = fs.readFileSync(req.files[1].path);

    const text1 = (await pdfParse(pdf1Buffer)).text;
    const text2 = (await pdfParse(pdf2Buffer)).text;

    // Word-level diff
    const changes = diff.diffWords(text1, text2);

    const result = changes.map(part => {
      return {
        added: part.added || false,
        removed: part.removed || false,
        value: part.value
      };
    });

    res.json({ success: true, diff: result });

    // Cleanup
    fs.unlinkSync(req.files[0].path);
    fs.unlinkSync(req.files[1].path);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Error comparing PDFs" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});

