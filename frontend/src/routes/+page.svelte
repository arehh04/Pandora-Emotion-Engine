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

    const API_URL = "https://arehhham-pandora.hf.space/predict";

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

    function exportToImage() {
        const element = document.getElementById('exportable-area');
        if (!element || typeof window === 'undefined') return;
        
        const btn = document.getElementById('pdf-btn');
        const originalText = btn!.innerText;
        btn!.innerText = "Capturing Image...";
        btn!.style.opacity = "0.7";
        btn!.style.pointerEvents = "none";

        // @ts-ignore
        html2canvas(element, { scale: 2, useCORS: true, backgroundColor: '#0a0f18' }).then((canvas) => {
            const link = document.createElement('a');
            link.download = 'Pandora_Insights_Profile.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            btn!.innerText = originalText;
            btn!.style.opacity = "1";
            btn!.style.pointerEvents = "auto";
        }).catch((err: any) => {
            console.error(err);
            btn!.innerText = "Error Capturing Image";
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
        if (val >= 0) return { bg: `rgba(0, 230, 118, ${0.1 + (intensity/255)*0.3})`, border: `rgba(0, 230, 118, ${0.4 + (intensity/255)*0.6})` };
        return { bg: `rgba(255, 82, 82, ${0.1 + (intensity/255)*0.3})`, border: `rgba(255, 82, 82, ${0.4 + (intensity/255)*0.6})` };
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
        <div class="dashboard-layout">
            
            <!-- Section 1: Genesis & The Dataset -->
            <div class="obsidian-panel">
                <h2 class="panel-header">1. Project Genesis & The Dataset</h2>
                <p class="narrative-text">
                    The core objective of the <strong>Pandora Emotion Engine</strong> is to predict a subject's Extraversion Quotient (EQ) strictly from raw, unstructured text. This eliminates the biases of traditional self-reported psychological surveys.
                </p>
                <div class="data-row">
                    <span class="data-label">Primary Source</span>
                    <span class="data-value">Hugging Face (bhadresh-savani/explore-emotion)</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Volume</span>
                    <span class="data-value">Tens of thousands of labeled records</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Preprocessing Pipeline</span>
                    <span class="data-value">Tokenization, Lemmatization, Punctuation Scrubbing</span>
                </div>
                <p class="narrative-text" style="margin-top: 1.5rem;">
                    We transformed the raw emotional classifications into a continuous Extraversion spectrum mapping by utilizing proven psychological frameworks, translating sparse labels into a dense numerical target (0 to 99).
                </p>
            </div>

            <!-- Section 2: Feature Engineering (Classical) -->
            <div class="obsidian-panel">
                <h2 class="panel-header">2. Feature Engineering (The Classical Era)</h2>
                <p class="narrative-text">
                    Before leaping into deep learning, we established a robust baseline. We hypothesized that Extraversion leaves specific linguistic footprints:
                </p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                    <div class="feature-card">
                        <div class="feature-title">Lexical Traces</div>
                        <div class="feature-desc">TF-IDF Vectorization projecting text into a 2,000-dimensional sparse space.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-title">Affect Lexicon</div>
                        <div class="feature-desc">The NRC Lexicon extracts 10 precise emotional vectors (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation, Positive, Negative).</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-title">Structural Syntax</div>
                        <div class="feature-desc">Part-of-Speech tagging measures the ratio of Nouns and Verbs, capturing interaction styles.</div>
                    </div>
                </div>

                <div class="data-row" style="border-bottom: none;">
                    <span class="data-label" style="color: #b0929c;">Baseline Model (Ridge Regression)</span>
                    <span class="data-value" style="color: #5e4652; font-size: 1.1rem;">R² 0.103</span>
                </div>
                <p class="narrative-text" style="font-size: 0.85rem; font-style: italic;">
                    While statistically significant, classical linear models struggled to capture the non-linear, contextual nuance of human emotion. We needed a heavier weapon.
                </p>
            </div>

            <!-- Section 3: Feature Fusion (Hybrid) -->
            <div class="obsidian-panel">
                <h2 class="panel-header">3. Feature Fusion (The Hybrid Approach)</h2>
                <p class="narrative-text">
                    We advanced to ensemble tree methods. To feed these models, we pioneered a "Feature Fusion" approach:
                </p>
                <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 4px; border: 1px dashed rgba(212,175,55,0.3); margin-bottom: 2rem; text-align: center;">
                    <span style="color: #d4af37; font-family: 'Cinzel'; font-weight: 700;">[ TF-IDF + NRC + POS ]</span>
                    <span style="color: #fff; margin: 0 1rem;">⊕</span>
                    <span style="color: #ff2a6d; font-family: 'Cinzel'; font-weight: 700;">[ 768-Dim Frozen BERT Embeddings ]</span>
                </div>

                <div class="svg-visualization" style="margin-bottom: 1rem;">
                    <svg viewBox="0 0 400 90" width="100%" height="90" xmlns="http://www.w3.org/2000/svg">
                        <text x="10" y="30" fill="#b0929c" font-size="12" font-family="Cinzel">XGBoost</text>
                        <rect x="130" y="20" width="180" height="12" fill="#8a2be2" rx="2"/>
                        <text x="320" y="30" fill="#e8d5db" font-size="12" font-family="Inter" font-weight="bold">R² 0.576</text>

                        <text x="10" y="70" fill="#b0929c" font-size="12" font-family="Cinzel">Random Forest</text>
                        <rect x="130" y="60" width="150" height="12" fill="#d4af37" rx="2" />
                        <text x="290" y="70" fill="#e8d5db" font-size="12" font-family="Inter" font-weight="bold">R² 0.473</text>
                    </svg>
                </div>
                <p class="narrative-text">
                    By feeding context-aware embeddings alongside classical features into non-linear decision trees, and aggressively tuning hyperparameters (max_depth=12, colsample_bytree=0.7), we achieved an unprecedented <strong>R² of 0.576</strong>. This ultimately became our State-of-the-Art model!
                </p>
            </div>

            <!-- Section 4: Deep Learning Zenith -->
            <div class="obsidian-panel">
                <h2 class="panel-header" style="color: #ff2a6d;">4. The Zenith: Fine-Tuned BERT</h2>
                <p class="narrative-text">
                    We abandoned feature extraction entirely. Instead, we constructed an end-to-end differentiable neural architecture. The entire BERT model was unfrozen and allowed to propagate gradients directly from the psychological target variable back to its core attention mechanisms.
                </p>
                
                <!-- The existing pipeline SVG -->
                <div style="margin: 3rem 0;">
                    <svg viewBox="0 0 430 120" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                        <rect x="0" y="45" width="80" height="30" rx="4" fill="rgba(255,255,255,0.05)" stroke="#b0929c" />
                        <text x="40" y="64" fill="#e8d5db" font-size="10" font-family="Inter" text-anchor="middle">Input Text</text>
                        <line x1="85" y1="60" x2="115" y2="60" stroke="#d4af37" stroke-dasharray="2,2" />
                        <polygon points="115,57 122,60 115,63" fill="#d4af37" />
                        
                        <rect x="125" y="40" width="100" height="40" rx="4" fill="rgba(255,42,109,0.1)" stroke="#ff2a6d" filter="url(#sigil-glow)" />
                        <text x="175" y="58" fill="#ff2a6d" font-size="11" font-weight="bold" font-family="Inter" text-anchor="middle">BERT Encoders</text>
                        <text x="175" y="72" fill="#b0929c" font-size="8" font-family="Inter" text-anchor="middle">12 Transformer Layers</text>
                        
                        <line x1="230" y1="60" x2="260" y2="60" stroke="#ff2a6d" />
                        <polygon points="260,57 267,60 260,63" fill="#ff2a6d" />
                        
                        <rect x="270" y="40" width="100" height="40" rx="4" fill="rgba(212,175,55,0.1)" stroke="#d4af37" />
                        <text x="320" y="58" fill="#d4af37" font-size="11" font-weight="bold" font-family="Inter" text-anchor="middle">Regression Head</text>
                        <text x="320" y="72" fill="#b0929c" font-size="8" font-family="Inter" text-anchor="middle">Dropout(0.3) → Linear(1)</text>
                        
                        <line x1="375" y1="60" x2="390" y2="60" stroke="#d4af37" />
                        <polygon points="390,57 397,60 390,63" fill="#d4af37" />
                        
                        <circle cx="410" cy="60" r="12" fill="#d4af37" filter="url(#sigil-glow)" />
                        <text x="410" y="64" fill="#000" font-size="10" font-weight="bold" font-family="Inter" text-anchor="middle">EQ</text>
                    </svg>
                </div>

                <div class="data-row" style="border-bottom: none;">
                    <span class="data-label" style="color: #ff2a6d; font-weight: bold;">Fine-Tuned BERT Performance</span>
                    <span class="data-value" style="color: #ff2a6d; font-size: 1.3rem; font-weight: 900; filter: drop-shadow(0 0 10px rgba(255,42,109,0.5));">R² 0.510</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Test Set Error</span>
                    <span class="data-value">RMSE: 16.42 | MAE: 12.85</span>
                </div>
                
                <p class="narrative-text" style="margin-top: 1.5rem;">
                    The architecture utilizes a final Sigmoid activation scaled by 99 to strictly bound predictions within the psychological test continuum. At Epoch 23, it achieved an incredible R² of 0.51, completely crushing early baselines.
                </p>
            </div>

            <!-- Section 5: XAI & Deployment -->
            <div class="obsidian-panel">
                <h2 class="panel-header">5. Explainability & Architecture</h2>
                <p class="narrative-text">
                    A black box model is useless in psychology. To establish trust, we integrated <strong>SHAP (SHapley Additive exPlanations)</strong> into the live inference pipeline. 
                </p>
                <div style="background: rgba(17,34,64,0.4); padding: 1.5rem; border-left: 4px solid #00e5ff; margin-bottom: 2rem;">
                    <p style="color: #e8d5db; font-size: 0.95rem; line-height: 1.6; font-family: 'Inter';">
                        SHAP calculates the exact marginal contribution of every single token in the input string towards the final Extraversion score. By color-coding these attributions (Green = Positive EQ influence, Red = Negative EQ influence), we provide full transparency into the model's psychological reasoning.
                    </p>
                </div>

                <div class="data-row">
                    <span class="data-label">Backend Deployment</span>
                    <span class="data-value">FastAPI + Docker + Redis + NGINX</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Caching Layer</span>
                    <span class="data-value">MD5 Hashing + Bloom Filter + Redis</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Frontend Framework</span>
                    <span class="data-value">SvelteKit + Vercel</span>
                </div>
            </div>

        </div>
    </main>
    {/if}

    <!-- TAB 2: LIVE DEMO -->
    {#if activeTab === 1}
    <main class="tab-content" in:fly={{y: 20, duration: 600}}>
        
        <div class="grid-2 grid-live-demo">
            
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
            <div class="diagnosis-container" style="flex-direction: column; padding: 2rem;">
                
                <!-- PROFILE HEADER -->
                <div class="profile-header" style="margin-bottom: 2.5rem; text-align: left;">
                    <div style="color:#00e5ff; font-size:1.2rem; letter-spacing:0.1em; text-transform:uppercase; font-family:'Cinzel', serif; font-weight:700;">
                        🧠 PROFILE: {analysisResult.persona.title.toUpperCase()} (Score: {analysisResult.score})
                    </div>
                    <div style="color:#b0929c; font-size:1rem; line-height:1.6; margin-top:1rem;">
                        Instead of a grade, think of <strong>{analysisResult.score}</strong> as a coordinate on a map of communication styles. 
                        Your linguistic footprint shows that you process the world internally before engaging with it. You don't waste words; you weigh them. While others rush to command the room, your model prediction suggests you prefer to <em>decode</em> the room. This isn't withdrawal—it's strategic data-gathering.
                    </div>
                </div>

                <!-- EMOTIONAL SIGNATURE -->
                <div class="emotional-signature" style="margin-bottom: 2.5rem; text-align: left; background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="color:#00e5ff; font-size:1.1rem; letter-spacing:0.1em; text-transform:uppercase; font-family:'Cinzel', serif; font-weight:700; margin-bottom:1rem;">
                        🔍 CURRENT EMOTIONAL SIGNATURE (Model Detections)
                    </div>
                    <div style="color:#b0929c; font-size:0.95rem; line-height:1.6;">
                        Our NLP classifier scanned your language for primary affective valences. Here is what the algorithm flagged in your recent text:
                        <ul style="margin-top:1rem; padding-left:0; list-style-type: none;">
                            {#if analysisResult.emotions.sadness > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(92,107,192,0.1); padding:0.8rem; border-radius:6px;"><strong>SADNESS:</strong> <span style="color:#5C6BC0; font-weight:bold;">PRESENT</span> → <em>The model detects a reflective, melancholic undertone. This often correlates with deep thinking, not necessarily distress—it signals you're processing complexity.</em></li>
                            {/if}
                            {#if analysisResult.emotions.fear > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(120,144,156,0.1); padding:0.8rem; border-radius:6px;"><strong>FEAR:</strong> <span style="color:#78909C; font-weight:bold;">PRESENT</span> → <em>The algorithm picked up on cautious or protective language. This suggests you are risk-aware and meticulous, rather than impulsive.</em></li>
                            {/if}
                            {#if analysisResult.emotions.anger > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(255,82,82,0.1); padding:0.8rem; border-radius:6px;"><strong>ANGER:</strong> <span style="color:#FF5252; font-weight:bold;">PRESENT</span> → <em>Markers of frustration or boundary-setting appeared. In this context, it usually translates to high standards or a desire for things to make logical sense.</em></li>
                            {/if}
                            {#if analysisResult.emotions.joy > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(212,175,55,0.1); padding:0.8rem; border-radius:6px;"><strong>JOY:</strong> <span style="color:#d4af37; font-weight:bold;">PRESENT</span> → <em>Markers of positivity and enthusiasm are evident. You project an uplifting and optimistic internal state.</em></li>
                            {/if}
                            {#if analysisResult.emotions.trust > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(0,230,118,0.1); padding:0.8rem; border-radius:6px;"><strong>TRUST:</strong> <span style="color:#00e676; font-weight:bold;">PRESENT</span> → <em>Words indicating safety and reliability were found. This signals a cooperative and open stance toward others.</em></li>
                            {/if}
                            {#if analysisResult.emotions.anticipation > 0}
                            <li style="margin-bottom:0.8rem; background:rgba(255,152,0,0.1); padding:0.8rem; border-radius:6px;"><strong>ANTICIPATION:</strong> <span style="color:#ff9800; font-weight:bold;">PRESENT</span> → <em>Forward-looking language detected, indicating planning and expectation.</em></li>
                            {/if}
                        </ul>
                        <div style="font-size:0.85rem; opacity:0.7; margin-top:1.5rem; font-style:italic;">
                            (⏱️ Inference Latency: {analysisResult.time_ms}ms – the time it took the transformer model to parse your text and compute these probabilities.)
                        </div>
                    </div>
                </div>

                <!-- BIG 5 RADAR -->
                <div class="big-5-section" style="margin-bottom: 2.5rem; display: flex; flex-direction: row; align-items: center; gap: 2rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 300px; text-align: left;">
                        <div style="color:#00e5ff; font-size:1.1rem; letter-spacing:0.1em; text-transform:uppercase; font-family:'Cinzel', serif; font-weight:700; margin-bottom:1rem;">
                            🧬 THE PSYCHOLOGICAL CONSTRUCT (Big 5)
                        </div>
                        <div style="color:#b0929c; font-size:0.95rem; line-height:1.7;">
                            Here are the core latent traits our model uses as the foundation for your profile:
                            <ul style="margin-top:0.8rem; padding-left:1.5rem;">
                                <li><strong>OPENNESS</strong> (High curiosity, abstract thinking)</li>
                                <li><strong>NEUROTICISM</strong> (Sensitivity to environmental stress)</li>
                                <li><strong>EXTRAVERSION</strong> (Energy drawn from social interaction)</li>
                                <li><strong>AGREEABLENESS</strong> (Tendency toward cooperation vs. competition)</li>
                                <li><strong>CONSCIENTIOUSNESS</strong> (Organization and discipline)</li>
                            </ul>
                            <blockquote style="border-left: 3px solid #ff2a6d; padding-left: 1rem; margin-top: 1.2rem; font-style:italic; background: rgba(255,42,109,0.05); padding: 0.8rem;">
                                Note: Your "{analysisResult.persona.title}" archetype is heavily influenced by your exact coordinates on these 5 dimensions—meaning you find energy in ideas rather than crowds.
                            </blockquote>
                        </div>
                    </div>
                    <div class="svg-visualization" style="flex: 1; min-width: 300px; display:flex; justify-content:center;">
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

                <!-- SHAP EXPLANATION -->
                <div class="shap-container" style="border: none; padding: 0; background: transparent; text-align: left;">
                    <div style="color:#00e5ff; font-size:1.1rem; letter-spacing:0.1em; text-transform:uppercase; font-family:'Cinzel', serif; font-weight:700; margin-bottom:1rem; display:flex; align-items:center;">
                        ⚖️ WHAT SHAPED YOUR SCORE? (SHAP Attributions)
                    </div>
                    
                    <div class="shap-description" style="text-align:left; background:transparent; border:none; padding:0; margin:0 0 1.5rem 0; font-size:0.95rem; color:#b0929c;">
                        To make our predictions transparent, we use <strong>SHAP</strong> (SHapley Additive exPlanations)—the industry standard for breaking down exactly <em>which</em> text features pushed your Extraversion score up or down.
                        <br/><br/>
                        Here is the feature influence breakdown for your specific input:
                    </div>
                    
                    <div class="shap-tokens-wrapper" style="margin-bottom: 1.5rem; justify-content: flex-start;">
                        {#if analysisResult.shap_tokens}
                            {#each analysisResult.shap_tokens as token}
                                {@const color = getShapColor(token.shap_value, Math.max(...analysisResult.shap_tokens.map((t: any) => Math.abs(t.shap_value))))}
                                <span class="shap-pill" title="SHAP: {token.shap_value.toFixed(4)}" 
                                      style="background: linear-gradient(135deg, {color.bg}, rgba(0,0,0,0.2)); border:1px solid {color.border}; box-shadow: 0 4px 15px {color.bg};">
                                    <span class="shap-token-text">{token.token}</span>
                                    <span class="shap-value-text" style="color: {token.shap_value > 0 ? '#00e676' : '#ff5252'}">
                                        {token.shap_value > 0 ? '+' : ''}{token.shap_value.toFixed(2)}
                                    </span>
                                </span>
                            {/each}
                        {/if}
                    </div>
                    
                    <div class="shap-footer" style="text-align:left; color:#b0929c; font-size:0.95rem; line-height:1.6; margin-top:0;">
                        <ul style="padding-left:1.5rem; margin-bottom:1.5rem;">
                            <li><strong style="color:#00e676;">Positive pushes (+)</strong> counterbalance you toward an outgoing structure.</li>
                            <li><strong style="color:#ff5252;">Negative pulls (-)</strong> pull your score downward, confirming a tendency for internal, solitary phrasing.</li>
                        </ul>
                        <strong>How to read this:</strong> Think of the ML model as a mirror—it's not judging you. It's simply highlighting the statistical weight of your word choices. Your position on the spectrum isn't a flaw; it's a computational signal of your unique style. Keep leaning into your analytical strength.
                    </div>
                </div>
            </div>
        </div> <!-- /exportable-area -->

        <div class="action-container">
            <button id="pdf-btn" class="ritual-btn" style="max-width: 400px; font-size: 0.9rem;" onclick={exportToImage}>Export Profile as Image</button>
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
    .dashboard-layout {
        display: flex;
        flex-direction: column;
        gap: 2.5rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    .narrative-text {
        color: #b0929c; 
        margin-bottom: 2rem; 
        font-weight: 300; 
        line-height: 1.8;
        font-size: 1.05rem;
    }
    .feature-card {
        background: rgba(0,0,0,0.4);
        border: 1px solid rgba(212,175,55,0.2);
        padding: 1.5rem;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #d4af37;
        box-shadow: 0 5px 15px rgba(212,175,55,0.1);
    }
    .feature-title {
        color: #d4af37;
        font-family: 'Cinzel', serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #e8d5db;
        font-size: 0.9rem;
        font-weight: 300;
        line-height: 1.5;
    }
    .grid-live-demo {
        grid-template-columns: 1.5fr 1fr;
    }
    @media (max-width: 900px) {
        .grid-2, .grid-live-demo { grid-template-columns: 1fr; gap: 2rem; }
        .obsidian-panel { padding: 1.5rem !important; }
        .container { padding: 1rem; }
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
        background: rgba(5, 1, 10, 0.75);
        backdrop-filter: blur(20px);
        z-index: 1000;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        overflow-y: auto;
        padding: 3rem 1rem;
    }
    
    .modal-overlay::before {
        content: '';
        position: fixed;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 70vw; height: 70vh;
        background: radial-gradient(circle, rgba(255,42,109,0.12) 0%, rgba(212,175,55,0.05) 40%, transparent 70%);
        border-radius: 50%;
        filter: blur(60px);
        pointer-events: none;
        z-index: -1;
    }
    
    .modal-content {
        width: 100%;
        max-width: 1000px;
        position: relative;
        animation: modalRise 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    
    @keyframes modalRise {
        from { opacity: 0; transform: translateY(30px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
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
        background: linear-gradient(145deg, rgba(20, 10, 25, 0.8) 0%, rgba(10, 5, 15, 0.95) 100%);
        backdrop-filter: blur(12px);
        padding: 4rem 3.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 30px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    
    #exportable-area::after {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='2' cy='2' r='1' fill='%23ffffff' fill-opacity='0.02'/%3E%3C/svg%3E");
        pointer-events: none;
    }
    
    #exportable-area::before {
        content: 'AI DIAGNOSTICS';
        position: absolute;
        top: 0; left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(90deg, transparent, rgba(212,175,55,0.2), transparent);
        border-bottom: 1px solid rgba(212,175,55,0.5);
        padding: 0.5rem 3rem;
        color: #d4af37;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.4em;
        border-radius: 0 0 8px 8px;
    }

    .diagnosis-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3rem;
    }
    @media (max-width: 768px) {
        .diagnosis-container { grid-template-columns: 1fr; gap: 2rem; }
        #exportable-area { padding: 1.2rem; }
        .nav-ritual { gap: 1rem; flex-wrap: wrap; text-align: center; }
        .score-value { font-size: 4rem; }
        .pandora-subtitle { font-size: 0.8rem; letter-spacing: 0.2em; text-align: center; }
        .action-container { flex-direction: column; align-items: center; }
        .panel-header { font-size: 1.2rem; }
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
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
        border: 1px solid;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        backdrop-filter: blur(5px);
    }
    .ember-gold { color: #FFD700; background: rgba(255, 215, 0, 0.1); border-color: rgba(255, 215, 0, 0.4); box-shadow: 0 0 10px rgba(255, 215, 0, 0.15); }
    .ember-orange { color: #FF9800; background: rgba(255, 152, 0, 0.1); border-color: rgba(255, 152, 0, 0.4); box-shadow: 0 0 10px rgba(255, 152, 0, 0.15); }
    .ember-green { color: #00E676; background: rgba(0, 230, 118, 0.1); border-color: rgba(0, 230, 118, 0.4); box-shadow: 0 0 10px rgba(0, 230, 118, 0.15); }

    .shap-container {
        margin-top: 3.5rem; 
        border-top: 1px solid rgba(255, 255, 255, 0.08); 
        padding-top: 2.5rem;
        position: relative;
        z-index: 10;
    }
    .shap-title {
        display: flex;
        align-items: center;
        justify-content: center;
        font-family:'Inter', sans-serif; 
        color:#00e5ff; 
        font-size: 1.1rem; 
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    .shap-description {
        color: #b0929c;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        line-height: 1.6;
        text-align: center;
        max-width: 700px;
        margin: 0 auto 2.5rem auto;
        background: rgba(0, 0, 0, 0.3);
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .shap-tokens-wrapper {
        display: flex; 
        flex-wrap: wrap; 
        justify-content: center; 
        gap: 0.8rem; 
        max-width: 850px; 
        margin: 0 auto;
    }
    .shap-pill {
        border-radius: 24px;
        padding: 6px 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #ffffff;
        backdrop-filter: blur(5px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: default;
    }
    .shap-pill:hover {
        transform: translateY(-2px) scale(1.02);
    }
    .shap-token-text {
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .shap-value-text {
        font-weight: 700;
        font-size: 0.8rem;
        background: rgba(0,0,0,0.3);
        padding: 2px 6px;
        border-radius: 10px;
    }
    .shap-footer {
        text-align: center; 
        margin-top: 2rem; 
        color: #8a7b82; 
        font-size: 0.85rem; 
        font-weight: 300;
    }

    .action-container {
        margin-top: 2rem;
        display: flex;
        justify-content: center;
        gap: 1.5rem;
    }
</style>
