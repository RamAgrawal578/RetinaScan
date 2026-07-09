/**
 * General page glue shared across all templates.
 */
document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss flash messages after a few seconds.
    document.querySelectorAll(".flash-container .alert").forEach((alertEl) => {
        setTimeout(() => {
            alertEl.style.transition = "opacity 0.4s ease";
            alertEl.style.opacity = "0";
            setTimeout(() => alertEl.remove(), 400);
        }, 6000);
    });
});
