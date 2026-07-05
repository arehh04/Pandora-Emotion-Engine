<script>
    import '../app.css';
    import { onMount } from 'svelte';
    import { browser } from '$app/environment';

    onMount(() => {
        if (!browser) return;

        // ── 24-second Dynamic Galaxy Canvas ──
        const canvas = document.getElementById('galaxy-canvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width, height;
        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        // Star and nebula parameters
        const TOTAL_STARS = 400;
        const stars = [];

        class Star {
            constructor() {
                this.reset(0); // start in Milky Way phase
            }
            reset(phaseOffset = 0) {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.baseRadius = Math.random() * 2.5 + 0.5;
                this.radius = this.baseRadius;
                // Each star gets a random phase offset within the 24s loop (0-24)
                this.phaseOffset = Math.random() * 24;
                // Individual twinkle speed
                this.twinkleSpeed = Math.random() * 0.02 + 0.005;
            }
        }

        for (let i = 0; i < TOTAL_STARS; i++) {
            stars.push(new Star());
        }

        // Phase definitions (total loop 24s)
        const PHASES = [
            { name: 'milkyway', start: 0, end: 8 },     // 0-8s
            { name: 'void', start: 8, end: 16 },        // 8-16s
            { name: 'nebula', start: 16, end: 24 }      // 16-24s
        ];

        function getCurrentPhase(time) {
            const t = time % 24;
            for (let p of PHASES) {
                if (t >= p.start && t < p.end) return p.name;
            }
            return 'milkyway'; // fallback
        }

        function getPhaseProgress(time) {
            const t = time % 24;
            const phase = PHASES.find(p => t >= p.start && t < p.end);
            if (!phase) return 0;
            return (t - phase.start) / (phase.end - phase.start);
        }

        const milkywayColors = [
            { r: 255, g: 200, b: 100, a: 0.9 },  // gold
            { r: 255, g: 180, b: 140, a: 0.8 },  // pinkish
            { r: 240, g: 160, b: 60, a: 0.9 },
            { r: 255, g: 220, b: 120, a: 0.95 }
        ];
        const voidColors = [
            { r: 100, g: 140, b: 220, a: 0.7 },  // blue
            { r: 130, g: 100, b: 200, a: 0.7 },  // purple
            { r: 70, g: 110, b: 190, a: 0.8 },
            { r: 180, g: 130, b: 210, a: 0.6 }
        ];
        const nebulaColors = [
            { r: 200, g: 50, b: 50, a: 0.8 },    // crimson
            { r: 180, g: 30, b: 80, a: 0.7 },
            { r: 220, g: 80, b: 40, a: 0.75 },
            { r: 255, g: 100, b: 50, a: 0.9 }
        ];

        function getStarColor(phase, brightness = 1) {
            let palette;
            if (phase === 'milkyway') palette = milkywayColors;
            else if (phase === 'void') palette = voidColors;
            else palette = nebulaColors;
            const idx = Math.floor(Math.random() * palette.length);
            const col = palette[idx];
            return `rgba(${col.r}, ${col.g}, ${col.b}, ${col.a * brightness})`;
        }

        function drawNebulaGlow(phase, progress) {
            let gradient;
            if (phase === 'milkyway') {
                gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, Math.max(width, height)*0.8);
                gradient.addColorStop(0, 'rgba(255, 200, 100, 0.15)');
                gradient.addColorStop(0.4, 'rgba(255, 160, 80, 0.08)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
            } else if (phase === 'void') {
                gradient = ctx.createRadialGradient(width*0.3, height*0.7, 0, width*0.5, height*0.5, Math.max(width,height));
                gradient.addColorStop(0, 'rgba(80, 100, 200, 0.08)');
                gradient.addColorStop(0.6, 'rgba(40, 50, 120, 0.04)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
            } else {
                gradient = ctx.createRadialGradient(width*0.7, height*0.4, 0, width*0.5, height*0.6, Math.max(width,height));
                gradient.addColorStop(0, 'rgba(200, 60, 50, 0.12)');
                gradient.addColorStop(0.5, 'rgba(150, 30, 40, 0.08)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
            }
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);
        }

        let animationFrameId;
        function animate(timestamp) {
            const time = timestamp / 1000;
            const phase = getCurrentPhase(time);
            const progress = getPhaseProgress(time);

            ctx.clearRect(0, 0, width, height);
            drawNebulaGlow(phase, progress);

            let zoom = 1;
            let panX = 0, panY = 0;
            if (phase === 'milkyway') {
                zoom = 1 + progress * 0.3;
                panX = -width * 0.1 * progress;
                panY = -height * 0.05 * progress;
            } else if (phase === 'void') {
                zoom = 1.3 - progress * 0.2;
                panX = width * 0.15 * progress;
                panY = height * 0.1 * progress;
            } else {
                zoom = 1.1 + progress * 0.4;
                panX = -width * 0.05 * progress;
                panY = -height * 0.08 * progress;
            }

            ctx.save();
            ctx.translate(width/2, height/2);
            ctx.scale(zoom, zoom);
            ctx.translate(-width/2 + panX, -height/2 + panY);

            for (let star of stars) {
                const starTime = (time + star.phaseOffset) % 24;
                const starPhase = getCurrentPhase(starTime);
                
                if (starPhase !== phase) continue;
                
                const twinkle = 0.6 + 0.4 * Math.sin(timestamp * star.twinkleSpeed + star.phaseOffset);
                const color = getStarColor(phase, twinkle);
                
                ctx.beginPath();
                ctx.arc(star.x, star.y, star.radius * twinkle, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                
                if (star.baseRadius > 1.8 && twinkle > 0.7) {
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.radius * 1.8 * twinkle, 0, Math.PI*2);
                    ctx.fillStyle = color.replace(/[\d\.]+\)$/, '0.3)'); 
                    ctx.fill();
                }
            }

            ctx.restore();
            animationFrameId = requestAnimationFrame(animate);
        }

        animationFrameId = requestAnimationFrame(animate);
        
        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationFrameId);
        }
    });
</script>

<!-- The Galaxy Canvas -->
<canvas id="galaxy-canvas"></canvas>

<!-- Subtle overlay layers -->
<div class="noise-overlay"></div>
<div class="ambient-glow"></div>

<slot />

<style>
    :global(*) {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    :global(html), :global(body) {
        font-family: 'Inter', sans-serif;
        background-color: transparent;
        color: #e8d5db;
        min-height: 100vh;
        overflow-x: hidden;
        line-height: 1.6;
    }

    /* ── The Dynamic Galaxy Canvas ── */
    #galaxy-canvas {
        position: fixed;
        top: 0; 
        left: 0; 
        width: 100vw; 
        height: 100vh;
        z-index: -2;
        background-color: #020005;
    }

    /* ── SVG Noise & Glow ── */
    .noise-overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        opacity: 0.15;
        z-index: -1;
        pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    }

    .ambient-glow {
        position: fixed;
        top: 20%; left: 50%;
        width: 80vw; height: 80vh;
        transform: translate(-50%, -50%);
        background: radial-gradient(ellipse, rgba(255, 42, 109, 0.03) 0%, rgba(212, 175, 55, 0.02) 40%, transparent 70%);
        z-index: -1;
        pointer-events: none;
        animation: pulseGlow 15s ease-in-out infinite alternate;
    }

    @keyframes pulseGlow {
        0% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
        100% { opacity: 1; transform: translate(-50%, -48%) scale(1.05); }
    }
</style>
