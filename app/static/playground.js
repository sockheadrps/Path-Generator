(function () {
    const stage = document.getElementById('stage');

    // Main Overlay
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');

    // Timeline Overlay
    const timelineCanvas = document.getElementById('timelineCanvas');
    const tCtx = timelineCanvas.getContext('2d');

    const startPoint = document.getElementById('startPoint');
    const targetPoint = document.getElementById('targetPoint');
    const pStatus = document.getElementById('pStatus');
    const pSteps = document.getElementById('pSteps');

    const randomizeBtn = document.getElementById('randomize');
    const generateBtn = document.getElementById('generate');


    // Parameter Inputs
    const params = [
        'speed', 'kp', 'stabilization', 'noise',
        'keep_prob_start', 'keep_prob_end', 'arc_strength', 'variance', 'overshoot_prob'
    ];
    const inputs = {};
    params.forEach(p => {
        inputs[p] = document.getElementById(p);
        if (inputs[p]) { // simplified check
            const valDisp = document.getElementById(p + 'Val');
            inputs[p].addEventListener('input', () => {
                if (valDisp) valDisp.textContent = inputs[p].value;
            });
        }
    });
    const realTimeAnimCheck = document.getElementById('realTimeAnim');

    let currentStart = null;
    let currentTarget = null;
    let isGenerating = false;

    // Chart Data
    let cumulativeDist = [];
    let totalDist = 0;

    // Path Data
    let lastPathA = null;

    function resizeCanvas() {
        overlay.width = stage.clientWidth;
        overlay.height = stage.clientHeight;

        timelineCanvas.width = timelineCanvas.clientWidth;
        timelineCanvas.height = timelineCanvas.clientHeight;

        // Redraw if data exists
        if (cumulativeDist.length > 0) renderTimeline(-1);
    }
    window.addEventListener('resize', resizeCanvas);

    // Wait for layout
    setTimeout(resizeCanvas, 100);

    function setStatus(t) { pStatus.textContent = t; }
    function setSteps(n) {
        // 500Hz simulation rate
        const ms = Math.round(n * (1000 / 500));
        pSteps.textContent = `Steps: ${n} (~${ms}ms)`;
    }

    function clearMainCanvas() { ctx.clearRect(0, 0, overlay.width, overlay.height); }
    function clearTimeline() { tCtx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height); }

    function randomizePoints() {
        const W = stage.clientWidth;
        const H = stage.clientHeight;
        const pad = 80;

        currentStart = {
            x: pad + Math.random() * (W - 2 * pad),
            y: pad + Math.random() * (H - 2 * pad)
        };
        currentTarget = {
            x: pad + Math.random() * (W - 2 * pad),
            y: pad + Math.random() * (H - 2 * pad)
        };

        updatePointIndicators();
        clearMainCanvas();
        clearTimeline();
        setStatus('Ready');
        setSteps(0);
        cumulativeDist = [];
        totalDist = 0;
        lastPathA = null;
    }

    function updatePointIndicators() {
        if (currentStart) {
            startPoint.style.left = currentStart.x + 'px';
            startPoint.style.top = currentStart.y + 'px';
            startPoint.classList.remove('hidden');
        }
        if (currentTarget) {
            targetPoint.style.left = currentTarget.x + 'px';
            targetPoint.style.top = currentTarget.y + 'px';
            targetPoint.classList.remove('hidden');
        }
    }

    async function generatePath() {
        if (!currentStart || !currentTarget || isGenerating) return;

        isGenerating = true;
        setStatus('Thinking…');
        generateBtn.disabled = true;
        clearMainCanvas();
        clearTimeline();

        const basePayload = {
            start: [currentStart.x, currentStart.y],
            target: [currentTarget.x, currentTarget.y],
            screen_w: stage.clientWidth,
            screen_h: stage.clientHeight,
        };
        // Add all params as floats/ints
        params.forEach(p => {
            if (inputs[p]) {
                const v = inputs[p].value;
                basePayload[p] = (p === 'max_steps') ? parseInt(v) : parseFloat(v);
            }
        });
        // Map unified params to generator expectations
        if (basePayload.kp !== undefined) {
            basePayload.kp_start = basePayload.kp;
            basePayload.kp_end = basePayload.kp;
            delete basePayload.kp;
        }

        try {
            const r = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(basePayload)
            });
            if (!r.ok) throw new Error(await r.text());
            const res = await r.json();

            // Store paths
            lastPathA = res.path;

            // Update Actuals display (Randomized values)
            if (res.actual_params) {
                const ap = res.actual_params;
                const setAct = (id, val, isPct = false) => {
                    const el = document.getElementById(id + 'Act');
                    if (el) el.textContent = ' (' + (isPct ? (val * 100).toFixed(0) + '%' : val.toFixed(3)) + ')';
                };

                setAct('speed', ap.speed);
                setAct('kp', ap.kp_start); // Use start as proxy for unified control
                setAct('keep_prob_start', ap.keep_prob_start, true);
                setAct('keep_prob_end', ap.keep_prob_end, true);
                setAct('arc_strength', ap.arc_strength);
                setAct('stabilization', ap.stabilization);
            }

            setSteps(res.steps);
            calculateProfiles(res.path);

            // 1. Clear Canvas
            clearMainCanvas();

            // 2. Animate Primary
            if (res.path) {
                animatePath(res.path);
            }

        } catch (err) {
            console.error(err);
            setStatus('Error: ' + err.message.substring(0, 20));
        } finally {
            isGenerating = false;
            generateBtn.disabled = false;
        }
    }

    function calculateProfiles(path) {
        if (!path || path.length < 2) return;

        cumulativeDist = [0];
        totalDist = 0;

        for (let i = 1; i < path.length; i++) {
            const step = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
            totalDist += step;
            cumulativeDist.push(totalDist);
        }
    }

    function renderTimeline(currentStepIdx) {
        const W = timelineCanvas.width;
        const H = timelineCanvas.height;
        tCtx.clearRect(0, 0, W, H);

        if (cumulativeDist.length < 2) return;

        // --- Draw Point Density (Rug Plot) ---
        // X-Axis = Cumulative Distance (0 -> Total Path Length)
        // Each tick = One point

        tCtx.beginPath();
        tCtx.strokeStyle = '#6aa9ff'; // Cyan
        tCtx.lineWidth = 1;
        tCtx.globalAlpha = 0.6; // Slight transparency to show overlaps (density)

        for (let i = 0; i < cumulativeDist.length; i++) {
            // Map distance to X coordinate
            // x = (curr / total) * W
            const x = (cumulativeDist[i] / totalDist) * W;

            // Draw vertical tick
            tCtx.moveTo(x, 15);
            tCtx.lineTo(x, H - 5);
        }
        tCtx.stroke();
        tCtx.globalAlpha = 1.0;

        // --- Draw Labels ---
        tCtx.font = '11px sans-serif';
        tCtx.fillStyle = '#888';
        tCtx.fillText('Start', 5, 12);
        tCtx.fillText('Target', W - 35, 12);

        tCtx.fillStyle = '#6aa9ff';
        tCtx.fillText('Point Density (Spatial)', W / 2 - 50, 12);

        // --- Draw Playhead ---
        if (currentStepIdx >= 0 && currentStepIdx < cumulativeDist.length) {
            const px = (cumulativeDist[currentStepIdx] / totalDist) * W;

            // Highlight current position
            tCtx.beginPath();
            tCtx.strokeStyle = '#fff';
            tCtx.lineWidth = 2;
            tCtx.moveTo(px, 0);
            tCtx.lineTo(px, H);
            tCtx.stroke();
        }
    }

    async function animatePath(path) {
        if (!path || path.length < 2) return;

        return new Promise((resolve) => {
            let lastFrameTime = performance.now();
            let i = 0;
            const hz = 500; // Physics simulation rate

            function step(now) {
                if (i >= path.length) {
                    renderTimeline(path.length - 1);
                    resolve();
                    return;
                }

                // Speed control
                const playbackSpeed = (realTimeAnimCheck && realTimeAnimCheck.checked) ? 1.0 : 3.0;

                // Time delta control
                const dt = now - lastFrameTime;
                lastFrameTime = now;

                // path points to consume = (dt / 1000) * hz * playbackSpeed
                let pointsToDraw = (dt / 1000) * hz * playbackSpeed;

                // Ensure we draw at least 1 if falling behind, but clamp max to avoid freeze
                if (pointsToDraw < 1) pointsToDraw = 1;
                if (pointsToDraw > 100) pointsToDraw = 100;

                // Render Timeline Sync
                renderTimeline(i);

                for (let k = 0; k < pointsToDraw && i < path.length; k++) {
                    ctx.beginPath();
                    ctx.lineWidth = 1.5;
                    ctx.strokeStyle = '#6aa9ff'; // Always Teal for animation
                    ctx.lineCap = 'round';

                    if (i === 0) {
                        ctx.moveTo(path[i][0], path[i][1]);
                    } else {
                        ctx.moveTo(path[i - 1][0], path[i - 1][1]);
                        ctx.lineTo(path[i][0], path[i][1]);
                    }
                    ctx.stroke();

                    // Optional: tiny dot at each point
                    if (i % 4 === 0) {
                        ctx.fillStyle = 'rgba(106, 169, 255, 0.4)';
                        ctx.beginPath();
                        ctx.arc(path[i][0], path[i][1], 1, 0, Math.PI * 2);
                        ctx.fill();
                    }

                    i++;
                }

                requestAnimationFrame(step);
            }
            requestAnimationFrame((t) => {
                lastFrameTime = t;
                step(t);
            });
        });
    }

    randomizeBtn.addEventListener('click', randomizePoints);
    generateBtn.addEventListener('click', generatePath);

    // --- Drag Logic ---
    let dragItem = null; // 'start' | 'target'

    function onMouseDown(e, item) {
        dragItem = item;
        e.preventDefault();
        e.stopPropagation();
    }

    // Bind mouse events for indicators
    startPoint.addEventListener('mousedown', (e) => onMouseDown(e, 'start'));
    targetPoint.addEventListener('mousedown', (e) => onMouseDown(e, 'target'));

    window.addEventListener('mouseup', () => { dragItem = null; });

    stage.addEventListener('mousemove', (e) => {
        if (!dragItem) return;

        const rect = stage.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Clamp to stage
        const cx = Math.max(0, Math.min(x, rect.width));
        const cy = Math.max(0, Math.min(y, rect.height));

        if (dragItem === 'start') {
            currentStart = { x: cx, y: cy };
        } else {
            currentTarget = { x: cx, y: cy };
        }

        updatePointIndicators();

        // Clear previous path on drag to avoid confusion
        clearMainCanvas();
        clearTimeline();
        setStatus('Ready');
        lastPathA = null;
    });

    // Check for Auto-Tuned settings
    try {
        const tuned = localStorage.getItem('pd_tune_settings');
        if (tuned) {
            const t = JSON.parse(tuned);
            // Map keys (let tuner decide variance/noise)
            const map = {
                'speed': t.speed,
                'kp': t.kp,
                'stabilization': t.stabilization,
                'arc_strength': t.arc_strength,
                'variance': t.variance || 0.0,
                'noise': t.noise || 0.0,
                'overshoot_prob': t.overshoot_prob || 0.0,
                'keep_prob_start': t.prob_start || 0.70,
                'keep_prob_end': t.prob_end || 0.98
            };

            Object.keys(map).forEach(k => {
                if (inputs[k]) {
                    inputs[k].value = map[k];
                    // Update display
                    const disp = document.getElementById(k + 'Val');
                    if (disp) disp.textContent = map[k];
                }
            });
            console.log("Applied tuned settings:", t);
            // Consume so it doesn't persist indefinitely
            localStorage.removeItem('pd_tune_settings');
        }
    } catch (e) {
        console.error("Failed to load tuned settings", e);
    }

    // Initial load
    randomizePoints();
})();
