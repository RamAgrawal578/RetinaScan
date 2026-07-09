/**
 * Lightweight scroll-reveal for section content. Respects
 * prefers-reduced-motion by skipping the animation entirely.
 */
(function () {
    const prefersReducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    document.addEventListener("DOMContentLoaded", function () {
        const targets = document.querySelectorAll(
            ".class-card, .flow-step, .info-card, .result-image-card"
        );
        if (!targets.length) return;

        if (prefersReducedMotion || !("IntersectionObserver" in window)) {
            targets.forEach((el) => el.classList.add("in-view"));
            return;
        }

        targets.forEach((el) => {
            el.style.opacity = "0";
            el.style.transform = "translateY(10px)";
            el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
        });

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = "1";
                        entry.target.style.transform = "translateY(0)";
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.15 }
        );

        targets.forEach((el) => observer.observe(el));
    });
})();
