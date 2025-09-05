const form = document.getElementById("uploadForm");
const resultDiv = document.getElementById("result");
const oldText = document.getElementById("oldText");
const newText = document.getElementById("newText");
const diffList = document.getElementById("diffList");
const dlMerged = document.getElementById("dl_merged");
const dlSide = document.getElementById("dl_side");
const dlSummary = document.getElementById("dl_summary");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultDiv.style.display = "none";
  diffList.innerHTML = "Working...";
  const fd = new FormData();
  fd.append("oldpdf", document.getElementById("oldpdf").files[0]);
  fd.append("newpdf", document.getElementById("newpdf").files[0]);

  try {
    const res = await fetch("/compare", { method: "POST", body: fd });
    if(!res.ok){
      const err = await res.json();
      diffList.innerHTML = "Error: " + (err.error||"server error");
      return;
    }
    const data = await res.json();
    // fill side-by-side preview
    resultDiv.style.display = "block";
    oldText.textContent = "";
    newText.textContent = "";
    diffList.innerHTML = "";

    // preview.pages is list of page diffs
    data.preview.pages.forEach(p=>{
      oldText.textContent += `--- Page ${p.page_index+1} ---\n` + (p.old_text||"") + "\n\n";
      newText.textContent += `--- Page ${p.page_index+1} ---\n` + (p.new_text||"") + "\n\n";
      p.diffs.forEach(d=>{
        const el = document.createElement("div");
        el.className = "diffRow";
        if(d.type === "insert"){
          el.innerHTML = `<b style="color:green">Inserted (page ${p.page_index+1}):</b> ${d.new}`;
        } else if(d.type === "delete"){
          el.innerHTML = `<b style="color:red">Removed (page ${p.page_index+1}):</b> ${d.old}`;
        } else {
          el.innerHTML = `<b style="color:orange">Changed (page ${p.page_index+1}):</b> <br/>OLD: ${d.old} <br/>NEW: ${d.new}`;
        }
        diffList.appendChild(el);
      });
    });

    dlMerged.href = "/output/" + data.annotated;
    dlSide.href = "/output/" + data.side_by_side;
    dlSummary.href = "/output/" + data.summary;

    // sync scroll between panels
    const oldPanel = document.getElementById("oldText");
    const newPanel = document.getElementById("newText");
    oldPanel.onscroll = ()=> { newPanel.scrollTop = oldPanel.scrollTop; };
    newPanel.onscroll = ()=> { oldPanel.scrollTop = newPanel.scrollTop; };

  } catch (err){
    diffList.innerHTML = "Unexpected error: "+err.message;
  }
});
