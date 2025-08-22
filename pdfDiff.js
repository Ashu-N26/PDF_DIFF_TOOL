document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = "Comparing...";

  try {
    const res = await fetch("/compare", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (!data.success) {
      resultDiv.innerHTML = "Error: " + (data.error || "Unknown");
      return;
    }

    // Render diff
    resultDiv.innerHTML = "";
    data.diff.forEach(part => {
      const span = document.createElement("span");
      span.textContent = part.value;
      if (part.added) span.classList.add("added");
      if (part.removed) span.classList.add("removed");
      resultDiv.appendChild(span);
    });
  } catch (err) {
    resultDiv.innerHTML = "Error comparing PDFs.";
    console.error(err);
  }
});
