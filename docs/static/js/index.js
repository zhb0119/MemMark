const copyButton = document.getElementById("copy-bibtex");
const bibtexCode = document.getElementById("bibtex-code");

async function copyBibTeX() {
  const citation = bibtexCode.innerText;

  try {
    await navigator.clipboard.writeText(citation);
  } catch (error) {
    const textarea = document.createElement("textarea");
    textarea.value = citation;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1800);
}

copyButton.addEventListener("click", copyBibTeX);
