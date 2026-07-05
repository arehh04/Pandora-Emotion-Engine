<script lang="ts">
    import { onMount } from 'svelte';
    import { fly, fade } from 'svelte/transition';

    let activeTab = $state(0);
    let inputText = $state("");
    let selectedModel = $state("Fine-Tuned BERT");
    
    let isModalOpen = $state(false);
    let isLoading = $state(false);
    let errorMsg = $state("");

    // The result from the API
    let analysisResult: any = $state(null);

    const API_URL = "http://localhost:8000/predict";

    async function commenceAnalysis() {
        if (!inputText.trim()) {
            errorMsg = "Please enter some text to analyze.";
            return;
        }
        errorMsg = "";
        isLoading = true;
        
        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: inputText, model: selectedModel })
            });
            if (!res.ok) throw new Error("API Request Failed");
            analysisResult = await res.json();
            
            // Open Modal
            isModalOpen = true;
            document.body.style.overflow = 'hidden';
        } catch (err: any) {
            errorMsg = err.message || "Failed to contact the API.";
        } finally {
            isLoading = false;
        }
    }

    function closeDiagnosisModal() {
        isModalOpen = false;
        document.body.style.overflow = 'auto';
    }

    function exportToPDF() {
        const element = document.getElementById('exportable-area');
        if (!element || typeof window === 'undefined') return;
        
        const btn = document.getElementById('pdf-btn');
        const originalText = btn!.innerText;
        btn!.innerText = "Forging PDF...";
        btn!.style.opacity = "0.7";
        btn!.style.pointerEvents = "none";

        const opt = {
            margin:       0.4,
            filename:     'Pandora_Diagnosis_Record.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2, useCORS: true, logging: false },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        // @ts-ignore
        html2pdf().set(opt).from(element).save().then(() => {
            btn!.innerText = originalText;
            btn!.style.opacity = "1";
            btn!.style.pointerEvents = "auto";
        }).catch((err: any) => {
            console.error(err);
            btn!.innerText = "Error Sealing Record";
            setTimeout(() => {
                btn!.innerText = originalText;
                btn!.style.opacity = "1";
                btn!.style.pointerEvents = "auto";
            }, 2000);
        });
    }

    // Dynamic SVG Radar logic
    function getPolygonPoints(radarData: any, radius: number = 100, center: number = 150) {
        const labels = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"];
        const angles = labels.map((_, i) => (i / labels.length) * 2 * Math.PI - Math.PI / 2); // Start at top
        
        let points = "";
        let circles = [];
        
        for (let i = 0; i < labels.length; i++) {
            const val = radarData[labels[i]] || 50;
            const r = (val / 100) * radius;
            const x = center + r * Math.cos(angles[i]);
            const y = center + r * Math.sin(angles[i]);
            points += `${x},${y} `;
            circles.push({x, y, label: labels[i]});
        }
        
        return { points: points.trim(), circles };
    }

    // Helper for color intensity in SHAP tokens
    function getShapColor(val: number, maxVal: number) {
        const intensity = Math.min(255, Math.floor(Math.abs(val) / (maxVal + 1e-8) * 255));
        if (val >= 0) return { bg: `rgba(0, ${intensity}, 100, 0.4)`, border: `rgba(0,229,100,0.6)` };
        return { bg: `rgba(${intensity}, 0, 0, 0.4)`, border: `rgba(255,80,80,0.6)` };
    }

</script>

<div class="container">
    <!-- Hero Section -->
    <header class="pandora-hero">
        <svg class="hero-sigil" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <filter id="sigil-glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <circle cx="50" cy="50" r="45" fill="none" stroke="#d4af37" stroke-width="1" opacity="0.3"/>
            <circle cx="50" cy="50" r="35" fill="none" stroke="#ff2a6d" stroke-width="1.5" opacity="0.5"/>
            <polygon points="50,10 85,75 15,75" fill="none" stroke="#d4af37" stroke-width="2" filter="url(#sigil-glow)"/>
            <polygon points="50,90 15,25 85,25" fill="none" stroke="#d4af37" stroke-width="2" filter="url(#sigil-glow)"/>
            <circle cx="50" cy="50" r="10" fill="#ff2a6d" opacity="0.8" filter="url(#sigil-glow)"/>
        </svg>

        <h1 class="pandora-title">PANDORA</h1>
        <div class="pandora-subtitle">Unlocking the Depths of the Human Psyche</div>
        <div class="pandora-divider"></div>
    </header>

    <!-- Navigation -->
    <nav class="nav-ritual">
        <button class="nav-btn {activeTab === 0 ? 'active' : ''}" onclick={() => activeTab = 0}>Methodology</button>
        <button class="nav-btn {activeTab === 1 ? 'active' : ''}" onclick={() => activeTab = 1}>Live Analysis</button>
    </nav>

    <!-- TAB 1: OVERVIEW -->
    {#if activeTab === 0}
    <main class="tab-content" in:fly={{y: 20, duration: 600}}>
        <div class="grid-2">
            
            <div class="obsidian-panel">
                <h2 class="panel-header">System Architecture</h2>
                <p style="color:#b0929c; margin-bottom: 2rem; font-weight: 300;">
                    Pandora is powered by a fusion of classical linguistic feature extraction and deep neural representations. We don't just count words; we understand context.
                </p>
                
                <div class="data-row">
                    <span class="data-label">Lexical Traces (TF-IDF)</span>
                    <span class="data-value">2,000 Dimensions</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Affect Lexicon (NRC)</span>
                    <span class="data-value">10 Emotional Vectors</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Structural Syntax</span>
                    <span class="data-value">Morphological Ratios</span>
                </div>
                <div class="data-row" style="border-bottom: none;">
                    <span class="data-label" style="color: #ff2a6d; font-weight: 500;">Neural Abstraction (BERT)</span>
                    <span class="data-value" style="color: #ff2a6d;">768 Contextual Dimensions</span>
                </div>
                
                <!-- End-to-End Pipeline Agentic Flow -->
                <div style="margin-top: 2rem;">
                    <h3 style="color:#d4af37; font-size: 0.9rem; font-family:'Cinzel', serif; letter-spacing:0.1em; margin-bottom: 1rem;">The End-to-End Pipeline</h3>
                    <svg viewBox="0 0 400 120" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                        <!-- User Input Node -->
                        <rect x="0" y="45" width="80" height="30" rx="4" fill="rgba(255,255,255,0.05)" stroke="#b0929c" />
                        <text x="40" y="64" fill="#e8d5db" font-size="10" font-family="Inter" text-anchor="middle">Input Text</text>
                        <!-- Arrow -->
                        <line x1="85" y1="60" x2="115" y2="60" stroke="#d4af37" stroke-dasharray="2,2" />
                        <polygon points="115,57 122,60 115,63" fill="#d4af37" />
                        
                        <!-- BERT Node -->
                        <rect x="125" y="40" width="100" height="40" rx="4" fill="rgba(255,42,109,0.1)" stroke="#ff2a6d" filter="url(#sigil-glow)" />
                        <text x="175" y="58" fill="#ff2a6d" font-size="11" font-weight="bold" font-family="Inter" text-anchor="middle">BERT Encoders</text>
                        <text x="175" y="72" fill="#b0929c" font-size="8" font-family="Inter" text-anchor="middle">12 Transformer Layers</text>
                        <!-- Arrow -->
                        <line x1="230" y1="60" x2="260" y2="60" stroke="#ff2a6d" />
                        <polygon points="260,57 267,60 260,63" fill="#ff2a6d" />
                        
                        <!-- Regression Head Node -->
                        <rect x="270" y="40" width="100" height="40" rx="4" fill="rgba(212,175,55,0.1)" stroke="#d4af37" />
                        <text x="320" y="58" fill="#d4af37" font-size="11" font-weight="bold" font-family="Inter" text-anchor="middle">Regression Head</text>
                        <text x="320" y="72" fill="#b0929c" font-size="8" font-family="Inter" text-anchor="middle">Dropout + Linear(1)</text>
                        <!-- Arrow -->
                        <line x1="375" y1="60" x2="390" y2="60" stroke="#d4af37" />
                        <polygon points="390,57 397,60 390,63" fill="#d4af37" />
                        
                        <!-- Output -->
                        <circle cx="410" cy="60" r="12" fill="#d4af37" filter="url(#sigil-glow)" />
                        <text x="410" y="64" fill="#000" font-size="10" font-weight="bold" font-family="Inter" text-anchor="middle">EQ</text>
                    </svg>
                </div>

            </div>

            <div class="obsidian-panel">
                <h2 class="panel-header">The Stress Test</h2>
                <p style="color:#b0929c; margin-bottom: 2rem; font-weight: 300;">
                    Rigorous evaluation against 3,210 unseen human records. End-to-end deep learning redefines the ceiling.
                </p>

                <!-- Custom SVG Performance Bar Chart -->
                <div class="svg-visualization" style="margin-bottom: 2rem;">
                    <svg viewBox="0 0 400 200" width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
                        <text x="10" y="30" fill="#b0929c" font-size="12" font-family="Cinzel">Ridge (Baseline)</text>
                        <rect x="130" y="20" width="80" height="12" fill="#5e4652" rx="2"/>
                        <text x="220" y="30" fill="#e8d5db" font-size="12" font-family="Inter">R² 0.103</text>

                        <text x="10" y="75" fill="#b0929c" font-size="12" font-family="Cinzel">XGBoost (Fusion)</text>
                        <rect x="130" y="65" width="120" height="12" fill="#8a2be2" rx="2"/>
                        <text x="260" y="75" fill="#e8d5db" font-size="12" font-family="Inter">R² 0.158</text>

                        <text x="10" y="120" fill="#b0929c" font-size="12" font-family="Cinzel">Random Forest</text>
                        <rect x="130" y="110" width="125" height="12" fill="#d4af37" rx="2" />
                        <text x="265" y="120" fill="#e8d5db" font-size="12" font-family="Inter">R² 0.162</text>
                        
                        <text x="10" y="165" fill="#ff2a6d" font-size="12" font-weight="bold" font-family="Cinzel">Fine-Tuned BERT ★</text>
                        <rect x="130" y="155" width="240" height="12" fill="url(#goldGrad)" rx="2" filter="url(#sigil-glow)"/>
                        <text x="380" y="165" fill="#ff2a6d" font-size="12" font-weight="bold" font-family="Inter">R² 0.311</text>

                        <defs>
                            <linearGradient id="goldGrad" x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stop-color="#ff2a6d" />
                                <stop offset="100%" stop-color="#d4af37" />
                            </linearGradient>
                        </defs>
                    </svg>
                </div>

                <div class="data-row">
                    <span class="data-label">Test Evaluation Set</span>
                    <span class="data-value">RMSE: 25.59 | MAE: 19.83</span>
                </div>
            </div>
        </div>
    </main>
    {/if}

    <!-- TAB 2: LIVE DEMO -->
    {#if activeTab === 1}
    <main class="tab-content" in:fly={{y: 20, duration: 600}}>
        
        <div class="grid-2" style="grid-template-columns: 1.5fr 1fr;">
            
            <div class="obsidian-panel">
                <h2 class="panel-header">Text Input</h2>
                <textarea class="incantation-box" bind:value={inputText} placeholder="Write about a recent interaction, your thought process, or social habits. For best results, use the first-person perspective."></textarea>
                
                <div style="margin: 1rem 0;">
                    <label style="color:#b0929c; font-size:0.9rem; font-family:'Cinzel', serif; margin-right: 1rem;">Engine:</label>
                    <select bind:value={selectedModel} style="background: rgba(0,0,0,0.5); color: #d4af37; border: 1px solid rgba(212, 175, 55, 0.3); padding: 0.5rem; font-family: 'Inter'; border-radius: 4px;">
                        <option value="Fine-Tuned BERT">Fine-Tuned BERT ★</option>
                        <option value="Ridge Regression">Ridge Regression (Baseline)</option>
                        <option value="XGBoost">XGBoost</option>
                        <option value="Random Forest">Random Forest</option>
                    </select>
                </div>

                {#if errorMsg}
                    <div style="color: #ff2a6d; margin-bottom: 1rem; font-size: 0.9rem;">{errorMsg}</div>
                {/if}
                
                <button class="ritual-btn" onclick={commenceAnalysis} disabled={isLoading}>
                    {#if isLoading}
                        Analyzing text...
                    {:else}
                        Analyze Persona
                    {/if}
                </button>
            </div>
            
            <div class="obsidian-panel">
                <h2 class="panel-header">Analysis Guidelines</h2>
                <div style="color:#d4af37; font-size:0.85rem; letter-spacing:0.1em; font-weight:700; margin-bottom:1rem; font-family:'Cinzel', serif;">HOW IT WORKS</div>
                
                <ul style="color:#b0929c; font-size:0.95rem; font-weight:300; line-height: 1.8; margin-left: 1.2rem; margin-bottom: 1.5rem;">
                    <li><strong>Perspective:</strong> Write in the first-person ("I", "me").</li>
                    <li><strong>Subject Matter:</strong> Describe your social habits, energy levels, and how you interact with others.</li>
                    <li><strong>Length:</strong> Aim for 3 to 5 sentences minimum to provide enough context for the engine.</li>
                    <li><strong>Model Selection:</strong> Choose the engine you want to evaluate. The Fine-Tuned BERT provides the most nuanced analysis.</li>
                </ul>
                
                <p style="color:#b0929c; margin-top: 1.5rem; font-weight: 300; font-size: 0.9rem; font-style: italic;">
                    Note: A longer, more authentic input allows the model to produce a more accurate psychological construct.
                </p>
            </div>
        </div>
    </main>
    {/if}
</div>

<!-- ── THE REVELATION MODAL ── -->
{#if isModalOpen && analysisResult}
<div id="revelationModal" class="modal-overlay active" transition:fade={{duration: 400}}>
    <div class="modal-content">
        <button class="close-seal" onclick={closeDiagnosisModal}>×</button>
        
        <!-- The area that will be converted to PDF -->
        <div id="exportable-area">
            <div class="diagnosis-container">
                
                <!-- Left: Score and Persona -->
                <div class="score-display">
                    <div style="color:#b0929c; font-size:0.8rem; letter-spacing:0.2em; text-transform:uppercase; margin-bottom: 0.5rem; font-family:'Cinzel', serif;">Extraversion Quotient</div>
                    <div class="score-value">{analysisResult.score}</div>
                    <div class="persona-title">{analysisResult.persona.title}</div>
                    <div class="persona-desc">{analysisResult.persona.desc}</div>
                    
                    <div style="margin-top:2rem;">
                        <div style="color:#d4af37; font-size:0.9rem; letter-spacing:0.15em; font-family:'Cinzel', serif; font-weight:700; margin-bottom:1rem;">Detected Traces</div>
                        <div>
                            {#if analysisResult.emotions.joy > 0} <span class="ember-tag ember-gold">Joy : Present</span> {/if}
                            {#if analysisResult.emotions.trust > 0} <span class="ember-tag ember-green">Trust : Present</span> {/if}
                            {#if analysisResult.emotions.anticipation > 0} <span class="ember-tag ember-orange">Anticipation : Present</span> {/if}
                            {#if analysisResult.emotions.sadness > 0} <span class="ember-tag" style="color: #5C6BC0; border-color: rgba(92,107,192,0.3)">Sadness : Present</span> {/if}
                            {#if analysisResult.emotions.fear > 0} <span class="ember-tag" style="color: #78909C; border-color: rgba(120,144,156,0.3)">Fear : Present</span> {/if}
                            {#if analysisResult.emotions.anger > 0} <span class="ember-tag" style="color: #FF5252; border-color: rgba(255,82,82,0.3)">Anger : Present</span> {/if}
                            <span class="ember-tag" style="color:#b0929c; border-color: rgba(255,255,255,0.1)">Time: {analysisResult.time_ms}ms</span>
                        </div>
                    </div>
                </div>

                <!-- Right: SVG Radar Chart -->
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1rem;">
                    <h3 style="color:#b0929c; font-size:0.9rem; letter-spacing:0.15em; font-family:'Cinzel', serif; margin-bottom:1rem; text-align:center;">The Psychological Construct</h3>
                    <div class="svg-visualization">
                        <svg viewBox="0 0 300 300" width="100%" height="250" xmlns="http://www.w3.org/2000/svg">
                            <g stroke="rgba(212,175,55,0.15)" stroke-width="1" fill="none">
                                <!-- Background Web -->
                                <polygon points="150,30 264,113 220,247 80,247 36,113" />
                                <polygon points="150,60 235,122 202,223 98,223 65,122" />
                                <polygon points="150,90 207,131 185,198 115,198 93,131" />
                                <polygon points="150,120 178,141 167,174 133,174 122,141" />
                                <line x1="150" y1="150" x2="150" y2="30" />
                                <line x1="150" y1="150" x2="264" y2="113" />
                                <line x1="150" y1="150" x2="220" y2="247" />
                                <line x1="150" y1="150" x2="80" y2="247" />
                                <line x1="150" y1="150" x2="36" y2="113" />
                            </g>
                            
                            <!-- Dynamic Polygon -->
                            {#if analysisResult.radar}
                                {@const radarSvg = getPolygonPoints(analysisResult.radar)}
                                <polygon points={radarSvg.points} fill="rgba(255,42,109,0.3)" stroke="#ff2a6d" stroke-width="2" filter="url(#sigil-glow)"/>
                                
                                <!-- Labels and Nodes -->
                                {#each radarSvg.circles as circle}
                                    <circle cx={circle.x} cy={circle.y} r="3" fill="#ff2a6d" />
                                    <text 
                                        x={circle.x + (circle.x > 150 ? 15 : (circle.x < 150 ? -15 : 0))} 
                                        y={circle.y + (circle.y > 150 ? 15 : (circle.y < 150 ? -15 : 0))} 
                                        fill={circle.label === 'Extraversion' ? '#d4af37' : '#b0929c'} 
                                        font-size="10" 
                                        font-family="Cinzel" 
                                        text-anchor={circle.x > 150 ? "start" : (circle.x < 150 ? "end" : "middle")}>
                                        {circle.label}
                                    </text>
                                {/each}
                            {/if}
                        </svg>
                    </div>
                </div>
            </div>

            <!-- Dynamic SHAP Waterfall / Tokens -->
            <div style="margin-top: 3rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 2rem;">
                <div style="text-align: center; font-family:'Cinzel', serif; color:#d4af37; font-size: 1.2rem; margin-bottom: 2rem;">
                    ⚖️ The Balance of Influence (SHAP Attributions)
                </div>
                
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; max-width: 800px; margin: 0 auto;">
                    {#if analysisResult.shap_tokens}
                        {#each analysisResult.shap_tokens as token}
                            {@const color = getShapColor(token.shap_value, Math.max(...analysisResult.shap_tokens.map((t: any) => Math.abs(t.shap_value))))}
                            <span 
                                title="SHAP: {token.shap_value.toFixed(4)}" 
                                style="background:{color.bg}; border:1px solid {color.border}; border-radius:4px; padding:4px 8px; display:inline-block; font-family:monospace; font-size:1rem; color:#E6F1FF;">
                                {token.token} <span style="font-size: 0.7rem; opacity: 0.7; margin-left: 4px;">({token.shap_value > 0 ? '+' : ''}{token.shap_value.toFixed(2)})</span>
                            </span>
                        {/each}
                    {/if}
                </div>
                <div style="text-align: center; margin-top: 1rem; color: #b0929c; font-size: 0.85rem; font-style: italic;">
                    Hover over tokens to see exact SHAP values. Green increases Extraversion, Red decreases it.
                </div>
            </div>
        </div> <!-- /exportable-area -->

        <div class="action-container">
            <button id="pdf-btn" class="ritual-btn" style="max-width: 400px; font-size: 0.9rem;" onclick={exportToPDF}>Export Results to PDF</button>
        </div>

    </div>
</div>
{/if}

<style>
    /* Scoped styling additions on top of global */
    .container {
        max-width: 1280px;
        margin: 0 auto;
        padding: 2rem;
        position: relative;
        z-index: 10;
    }

    .pandora-hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .hero-sigil {
        width: 60px;
        height: 60px;
        margin-bottom: 1.5rem;
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(2deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    .pandora-title {
        font-family: 'Cinzel', serif;
        font-size: clamp(3.5rem, 8vw, 6.5rem);
        font-weight: 900;
        letter-spacing: 0.15em;
        background: linear-gradient(180deg, #ffffff 0%, #d4af37 50%, #ff2a6d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 10px 40px rgba(212, 175, 55, 0.2);
        line-height: 1;
        margin-bottom: 1rem;
    }

    .pandora-subtitle {
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        color: #b0929c;
        letter-spacing: 0.4em;
        text-transform: uppercase;
        font-weight: 600;
    }

    .pandora-divider {
        width: 2px;
        height: 60px;
        background: linear-gradient(180deg, transparent, #d4af37, #ff2a6d, transparent);
        margin: 2rem auto;
        opacity: 0.6;
    }

    .nav-ritual {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 3rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.15);
        padding-bottom: 1rem;
    }
    
    .nav-btn {
        background: transparent;
        border: none;
        color: #7a656d;
        font-family: 'Cinzel', serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        cursor: pointer;
        position: relative;
        padding: 0.5rem 1rem;
        transition: color 0.4s ease;
    }
    
    .nav-btn:hover { color: #d4af37; }
    .nav-btn.active { color: #d4af37; }
    
    .nav-btn::after {
        content: '';
        position: absolute;
        bottom: -17px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 2px;
        background: #ff2a6d;
        transition: width 0.4s ease, box-shadow 0.4s ease;
    }
    
    .nav-btn.active::after {
        width: 100%;
        box-shadow: 0 0 10px #ff2a6d;
    }

    .tab-content {
        /* transitions handled by svelte transition directives */
    }

    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
    }
    @media (max-width: 900px) {
        .grid-2 { grid-template-columns: 1fr; gap: 2rem; }
    }

    .obsidian-panel {
        background: linear-gradient(145deg, rgba(15, 4, 11, 0.8) 0%, rgba(5, 1, 10, 0.9) 100%);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 4px;
        padding: 2.5rem;
        position: relative;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
    }
    
    .obsidian-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='2' cy='2' r='1' fill='%23d4af37' fill-opacity='0.03'/%3E%3C/svg%3E");
        pointer-events: none;
    }

    .panel-header {
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #d4af37;
        letter-spacing: 0.1em;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .panel-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.3), transparent);
    }

    .incantation-box {
        width: 100%;
        height: 200px;
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(138, 43, 226, 0.4);
        border-radius: 2px;
        color: #f7eef1;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 300;
        line-height: 1.8;
        padding: 1.5rem;
        transition: all 0.4s ease;
        resize: vertical;
    }
    .incantation-box:focus {
        outline: none;
        border-color: #ff2a6d;
        background: rgba(20, 2, 8, 0.6);
        box-shadow: inset 0 0 20px rgba(255, 42, 109, 0.1);
    }
    .incantation-box::placeholder {
        color: #5e4652;
        font-style: italic;
    }
    
    .ritual-btn {
        background: linear-gradient(90deg, #1f0714, #2a0515);
        color: #d4af37;
        border: 1px solid rgba(212, 175, 55, 0.5);
        border-radius: 2px;
        font-family: 'Cinzel', serif;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.15em;
        padding: 1rem 2rem;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        cursor: pointer;
        width: 100%;
        display: block;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
    }
    .ritual-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .ritual-btn::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.2), transparent);
        transition: left 0.6s ease;
    }
    
    .ritual-btn:hover:not(:disabled) {
        border-color: #d4af37;
        background: #2a0515;
        color: #fff;
        box-shadow: 0 0 30px rgba(255, 42, 109, 0.2);
        transform: translateY(-2px);
    }
    .ritual-btn:hover:not(:disabled)::before { left: 100%; }

    /* Modal Overlay */
    .modal-overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(5, 1, 10, 0.85);
        backdrop-filter: blur(10px);
        z-index: 1000;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        overflow-y: auto;
        padding: 3rem 1rem;
    }
    
    .modal-content {
        width: 100%;
        max-width: 1000px;
        position: relative;
    }

    .close-seal {
        position: absolute;
        top: 1.5rem; right: 1.5rem;
        background: none; border: none;
        color: #b0929c;
        font-size: 2rem;
        cursor: pointer;
        z-index: 20;
        transition: color 0.3s ease, transform 0.3s ease;
    }
    .close-seal:hover {
        color: #ff2a6d;
        transform: scale(1.1) rotate(90deg);
    }

    #exportable-area {
        background-color: #0c0310; 
        padding: 3rem;
        border-radius: 4px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        position: relative;
    }
    
    #exportable-area::before {
        content: 'THE TRUTH UNSEALED';
        position: absolute;
        top: -10px; left: 50%;
        transform: translateX(-50%);
        background: #0c0310;
        padding: 0 1rem;
        color: #ff2a6d;
        font-family: 'Cinzel', serif;
        font-size: 0.8rem;
        letter-spacing: 0.4em;
    }

    .diagnosis-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
    }
    @media (max-width: 768px) {
        .diagnosis-container { grid-template-columns: 1fr; }
        #exportable-area { padding: 1.5rem; }
    }

    .score-display {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .score-value {
        font-family: 'Cinzel', serif;
        font-size: 5.5rem;
        font-weight: 900;
        line-height: 1;
        background: linear-gradient(135deg, #d4af37 0%, #ff2a6d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 20px rgba(255, 42, 109, 0.2));
    }

    .persona-title {
        font-family: 'Cinzel', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #d4af37;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    .persona-desc {
        color: #e8d5db;
        font-size: 0.95rem;
        line-height: 1.8;
        font-weight: 300;
    }

    .svg-visualization {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
    }

    .data-row { 
        display: flex; 
        justify-content: space-between; 
        padding: 12px 0; 
        border-bottom: 1px solid rgba(255,255,255,0.05); 
    }
    .data-label { color: #b0929c; font-size: 0.9rem; font-weight: 300; }
    .data-value { color: #d4af37; font-weight: 500; font-size: 0.9rem; font-family: 'Inter', monospace; }

    .ember-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 2px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 4px 4px 4px 0;
        border: 1px solid;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: 'Cinzel', serif;
    }
    .ember-gold { color: #FFD700; background: rgba(255, 215, 0, 0.05); border-color: rgba(255, 215, 0, 0.3); }
    .ember-orange { color: #FF9800; background: rgba(255, 152, 0, 0.05); border-color: rgba(255, 152, 0, 0.3); }
    .ember-green { color: #00E676; background: rgba(0, 230, 118, 0.05); border-color: rgba(0, 230, 118, 0.3); }

    .action-container {
        margin-top: 2rem;
        display: flex;
        justify-content: center;
        gap: 1.5rem;
    }
</style>
