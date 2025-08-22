const pdfjsLib = window["pdfjs-dist/build/pdf"];
pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const form = document.getElementById("compareForm");
const canvasOld = document.getElementById("canvasOld");
const canvasNew = document.getElementById("canvasNew");
const ctxOld = canvasOld.getContext("2d");
const ctxNew = canvasNew.getContext("2d");
const resultLinks = document.getElementById("resultLinks");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultLinks.innerHTML = "Comparing…";

  const fd = new FormData(form);
  const resp = await fetch("/api/compare", { method: "POST", body: fd });
  const data = await resp.json();
  if (!resp.ok) {
    resultLinks.innerHTML = `<span style="color:#b00">Error: ${data.error || "failed"}</span>`;
    return;
  }

  const { mergedPdfUrl, diffJsonUrl } = data;
  resultLinks.innerHTML = `
    <a href="${mergedPdfUrl}" target="_blank">Download merged PDF</a>
    &nbsp;|&nbsp;
    <a href="${diffJsonUrl}" target="_blank">View diff JSON</a>
    <span class="badge add">Added: ${data.summary.counts.added}</span>
    <span class="badge rem">Removed: ${data.summary.counts.removed}</span>
    <span class="badge chg">Changed: ${data.summary.counts.changed}</span>
  `;

  // Load OLD & NEW to canvases (first page) + draw overlays for NEW using diff JSON.
  // (This is a quick preview; full detail is in merged PDF.)
  await renderSideBySide(fd.get("pdf1"), fd.get("pdf2"), diffJsonUrl);
});

async function renderSideBySide(oldFile, newFile, diffJsonUrl) {
  // read as arrayBuffer
  const oldBuf = await oldFile.arrayBuffer();
  const newBuf = await newFile.arrayBuffer();

  const oldPdf = await pdfjsLib.getDocument({ data: oldBuf }).promise;
  const newPdf = await pdfjsLib.getDocument({ data: newBuf }).promise;

  const oldPage = await oldPdf.getPage(1);
  const newPage = await newPdf.getPage(1);

  const scale = 1.2;
  const vOld = oldPage.getViewport({ scale });
  const vNew = newPage.getViewport({ scale });

  canvasOld.width = vOld.width; canvasOld.height = vOld.height;
  canvasNew.width = vNew.width; canvasNew.height = vNew.height;

  await oldPage.render({ canvasContext: ctxOld, viewport: vOld }).promise;
  await newPage.render({ canvasContext: ctxNew, viewport: vNew }).promise;

  // fetch diff and overlay first-page boxes
  const diff = await (await fetch(diffJsonUrl)).json();
  const p0 = diff.pages.find(p => p.pageIndex === 0);
  if (!p0) return;

  ctxNew.lineWidth = 2;
  p0.changes.forEach(ch => {
    if (ch.type === "added") ctxNew.strokeStyle = "rgba(0,150,0,0.9)";
    else ctxNew.strokeStyle = "rgba(200,0,0,0.95)";

    // scale boxes from PDF units to canvas units
    const sx = vNew.width / p0.width;
    const sy = vNew.height / p0.height;
    ctxNew.strokeRect(ch.box.x * sx, ch.box.y * sy, ch.box.w * sx, ch.box.h * sy);
  });
}
