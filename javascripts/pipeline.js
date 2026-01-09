// Interactive Pipeline Diagram
document.addEventListener('DOMContentLoaded', function () {
    initPipelineDiagram();
});

// Also reinit on navigation (for mkdocs instant loading)
if (typeof document$ !== 'undefined') {
    document$.subscribe(function () {
        initPipelineDiagram();
    });
}

function initPipelineDiagram() {
    const steps = document.querySelectorAll('.pipeline-step');
    const details = document.querySelectorAll('.detail-content');
    const placeholder = document.querySelector('.pipeline-placeholder');

    if (steps.length === 0) return;

    steps.forEach(step => {
        step.addEventListener('click', function () {
            const targetId = this.dataset.target;

            // Update active step
            steps.forEach(s => s.classList.remove('active'));
            this.classList.add('active');

            // Hide placeholder
            if (placeholder) {
                placeholder.style.display = 'none';
            }

            // Show corresponding detail
            details.forEach(d => d.classList.remove('active'));
            const targetDetail = document.getElementById(targetId);
            if (targetDetail) {
                targetDetail.classList.add('active');
            }
        });
    });
}
