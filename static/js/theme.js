/**
 * Dark / light theme toggle. Persists the choice in localStorage and
 * respects the OS-level preference on first visit.
 */
(function () {
    const STORAGE_KEY = "retinaai-theme";
    const root = document.documentElement;

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
    }

    function getPreferredTheme() {
        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (stored === "light" || stored === "dark") return stored;
        return window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";
    }

    applyTheme(getPreferredTheme());

    document.addEventListener("DOMContentLoaded", function () {
        const toggleBtn = document.getElementById("themeToggle");
        if (!toggleBtn) return;

        toggleBtn.addEventListener("click", function () {
            const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
            const next = current === "dark" ? "light" : "dark";
            applyTheme(next);
            window.localStorage.setItem(STORAGE_KEY, next);
        });
    });
})();
