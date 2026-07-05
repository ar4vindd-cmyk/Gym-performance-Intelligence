document.addEventListener("DOMContentLoaded", () => {
    const result = document.querySelector("#result");
    if (result) {
        result.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    const form = document.querySelector("form");
    const button = form?.querySelector("button");
    form?.addEventListener("submit", () => {
        if (button) {
            button.disabled = true;
            button.textContent = "Training models and generating report...";
        }
    });
});
