// MD Trajectory Visualization - Main JavaScript
// This file is rendered as a Jinja2 template

// Global state
const state = {
    currentTimescaleIndex: 0,
    selectedFrameIndex: null,    // index into prep/timescale embeddings
    selectedPrepState: null,     // prep cluster label of selected frame
    selectedVampState: null,     // VAMP state of selected frame (on current timescale)
    selectedAttentionState: null,// state shown in attention viewer
    showAttention: true,
    protein: {
        representation: 'cartoon'
    }
};

// D3.js embedding plot
let embeddingSvg, embeddingG, xScale, yScale, embeddingZoom;
const embeddingMargin = { top: 20, right: 20, bottom: 40, left: 50 };

// D3.js alluvial
let alluvialSvg;

// D3.js matrix
let matrixSvg;

// 3Dmol.js viewers
let proteinViewer, attentionViewer;

// Determine data source for 2D scatter
const hasPrep = !!(VISUALIZATION_DATA.prep && VISUALIZATION_DATA.prep.embeddings);

// Initialize visualization
function init() {
    console.log('Initializing MD Trajectory Visualization...');
    console.log(`Loaded ${VISUALIZATION_DATA.timescales.length} timescales`);
    if (hasPrep) {
        console.log(`Prep data: ${VISUALIZATION_DATA.prep.n_frames} frames, ${VISUALIZATION_DATA.prep.n_states} clusters`);
    }

    const loading = document.getElementById('loading');

    try {
        // Apply theme
        if (VISUALIZATION_DATA.config.theme === 'light') {
            document.body.classList.add('light-theme');
        }

        // Initialize all components
        initTimescaleControls();
        initEmbeddingPlot();
        initAlluvialPlot();
        initProteinViewer();
        initTransitionMatrix();
        initDiagnosticsPanel();
        initEventListeners();

        // Load first timescale
        loadTimescale(0);

        // Hide loading screen on success
        if (loading) loading.style.display = 'none';
        console.log('Initialization complete');
    } catch (e) {
        console.error('Initialization error:', e);
        // Replace loading spinner with error message
        if (loading) {
            loading.innerHTML = `
                <p style="color: #ff6b6b; font-weight: bold;">Initialization Error</p>
                <p style="color: #ccc; font-size: 12px; margin-top: 8px; white-space: pre-wrap; text-align: left; max-width: 600px;">${e.message}\n\n${e.stack || ''}</p>
            `;
        }
    }
}

// =============================================================================
// Timescale controls (sidebar buttons)
// =============================================================================

function initTimescaleControls() {
    const container = document.getElementById('timescale-list');
    if (!container) return;
    container.innerHTML = '';

    VISUALIZATION_DATA.timescales.forEach((ts, index) => {
        const btn = document.createElement('button');
        btn.className = 'timescale-btn';
        if (index === 0) btn.classList.add('active');

        const label = document.createElement('span');
        label.className = 'timescale-label';
        label.textContent = `Lag ${ts.lagtime}`;
        btn.appendChild(label);

        const details = document.createElement('span');
        details.className = 'timescale-details';
        details.textContent = `${ts.n_frames} frames, ${ts.n_states} states`;
        btn.appendChild(details);

        btn.addEventListener('click', () => {
            document.querySelectorAll('.timescale-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTimescale(index);
        });

        container.appendChild(btn);
    });
}

// =============================================================================
// Embedding scatter plot (uses prep data if available, else first timescale)
// =============================================================================

function getEmbeddingData() {
    if (hasPrep) {
        return {
            embeddings: VISUALIZATION_DATA.prep.embeddings,
            labels: VISUALIZATION_DATA.prep.cluster_labels,
            bounds: VISUALIZATION_DATA.prep_bounds,
            n_frames: VISUALIZATION_DATA.prep.n_frames
        };
    }
    // Fallback: use first timescale embeddings
    const ts = VISUALIZATION_DATA.timescales[0];
    return {
        embeddings: ts.embeddings,
        labels: ts.state_assignments,
        bounds: VISUALIZATION_DATA.bounds,
        n_frames: ts.n_frames
    };
}

function initEmbeddingPlot() {
    const container = document.getElementById('embedding-plot');
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const embData = getEmbeddingData();
    const bounds = embData.bounds;

    // Add padding to bounds
    const padX = (bounds.max_x - bounds.min_x) * 0.05;
    const padY = (bounds.max_y - bounds.min_y) * 0.05;

    // Scales
    xScale = d3.scaleLinear()
        .domain([bounds.min_x - padX, bounds.max_x + padX])
        .range([embeddingMargin.left, width - embeddingMargin.right]);

    yScale = d3.scaleLinear()
        .domain([bounds.min_y - padY, bounds.max_y + padY])
        .range([height - embeddingMargin.bottom, embeddingMargin.top]);

    // SVG
    embeddingSvg = d3.select('#embedding-plot')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Clip path for points
    embeddingSvg.append('defs').append('clipPath')
        .attr('id', 'plot-clip')
        .append('rect')
        .attr('x', embeddingMargin.left)
        .attr('y', embeddingMargin.top)
        .attr('width', width - embeddingMargin.left - embeddingMargin.right)
        .attr('height', height - embeddingMargin.top - embeddingMargin.bottom);

    // Gridlines
    embeddingSvg.append('g')
        .attr('class', 'grid x-grid')
        .attr('transform', `translate(0,${height - embeddingMargin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(8)
            .tickSize(-(height - embeddingMargin.top - embeddingMargin.bottom))
            .tickFormat(''));

    embeddingSvg.append('g')
        .attr('class', 'grid y-grid')
        .attr('transform', `translate(${embeddingMargin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(8)
            .tickSize(-(width - embeddingMargin.left - embeddingMargin.right))
            .tickFormat(''));

    // Axes
    embeddingSvg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - embeddingMargin.bottom})`)
        .call(d3.axisBottom(xScale).ticks(8));

    embeddingSvg.append('g')
        .attr('class', 'y-axis')
        .attr('transform', `translate(${embeddingMargin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(8));

    // Points group (clipped)
    embeddingG = embeddingSvg.append('g')
        .attr('clip-path', 'url(#plot-clip)');

    // Zoom behavior
    embeddingZoom = d3.zoom()
        .scaleExtent([0.5, 20])
        .on('zoom', onEmbeddingZoom);

    embeddingSvg.call(embeddingZoom);

    // Draw embedding points (drawn once, constant across timescales)
    drawEmbeddingPoints();

    // Resize handler
    window.addEventListener('resize', onWindowResize);
}

function drawEmbeddingPoints() {
    if (!embeddingG) return;

    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const embData = getEmbeddingData();

    // Build data array
    const data = embData.embeddings.map((point, i) => ({
        x: point[0],
        y: point[1],
        index: i,
        clusterLabel: embData.labels[i]
    }));

    // Draw points
    embeddingG.selectAll('.embedding-point')
        .data(data, d => d.index)
        .enter()
        .append('circle')
        .attr('class', 'embedding-point')
        .attr('r', 4)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('fill', d => stateColors[d.clusterLabel % stateColors.length])
        .on('mouseover', function(event, d) {
            if (!d3.select(this).classed('selected')) {
                d3.select(this).attr('r', 6);
            }
            // Look up VAMP state for current timescale
            const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
            const vampState = (d.index < ts.state_assignments.length)
                ? ts.state_assignments[d.index] : '?';
            showTooltip(event, `
                Frame: ${d.index}<br/>
                Prep State: ${d.clusterLabel}<br/>
                VAMP State: ${vampState}<br/>
                X: ${d.x.toFixed(3)}<br/>
                Y: ${d.y.toFixed(3)}
            `);
        })
        .on('mousemove', function(event) {
            const tooltip = document.getElementById('tooltip');
            if (tooltip) {
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY + 10) + 'px';
            }
        })
        .on('mouseout', function() {
            if (!d3.select(this).classed('selected')) {
                d3.select(this).attr('r', 4);
            }
            hideTooltip();
        })
        .on('click', function(event, d) {
            onFrameClick(d, this);
        });
}

function onFrameClick(d, element) {
    // Deselect previous
    embeddingG.selectAll('.embedding-point.selected')
        .classed('selected', false)
        .attr('r', 4);

    // Select this point
    d3.select(element).classed('selected', true).attr('r', 6);

    state.selectedFrameIndex = d.index;
    state.selectedPrepState = d.clusterLabel;

    // Look up VAMP state for current timescale
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    state.selectedVampState = (d.index < ts.state_assignments.length)
        ? ts.state_assignments[d.index] : null;

    updateAlluvialPlot();
    updateProteinViewer();

    console.log(`Selected frame ${d.index}: prep=${d.clusterLabel}, vamp=${state.selectedVampState}`);
}

// Zoom handler
function onEmbeddingZoom(event) {
    const transform = event.transform;
    const newXScale = transform.rescaleX(xScale);
    const newYScale = transform.rescaleY(yScale);

    const container = document.getElementById('embedding-plot');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Update axes
    embeddingSvg.select('.x-axis')
        .call(d3.axisBottom(newXScale).ticks(8));
    embeddingSvg.select('.y-axis')
        .call(d3.axisLeft(newYScale).ticks(8));

    // Update gridlines
    embeddingSvg.select('.x-grid')
        .call(d3.axisBottom(newXScale).ticks(8)
            .tickSize(-(height - embeddingMargin.top - embeddingMargin.bottom))
            .tickFormat(''));
    embeddingSvg.select('.y-grid')
        .call(d3.axisLeft(newYScale).ticks(8)
            .tickSize(-(width - embeddingMargin.left - embeddingMargin.right))
            .tickFormat(''));

    // Update points
    embeddingG.selectAll('.embedding-point')
        .attr('cx', d => newXScale(d.x))
        .attr('cy', d => newYScale(d.y));
}

// =============================================================================
// Alluvial diagram
// =============================================================================

function initAlluvialPlot() {
    const container = document.getElementById('alluvial-plot');
    if (!container) return;

    // Show placeholder
    container.innerHTML = '<div class="alluvial-placeholder">Click a point in the embeddings<br/>to see state transitions.</div>';
}

function updateAlluvialPlot() {
    const container = document.getElementById('alluvial-plot');
    if (!container) return;

    // Clear previous
    container.innerHTML = '';

    if (state.selectedPrepState === null) {
        container.innerHTML = '<div class="alluvial-placeholder">Click a point in the embeddings<br/>to see prep → VAMP state mapping.</div>';
        return;
    }

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const prepLabels = hasPrep ? VISUALIZATION_DATA.prep.cluster_labels : null;
    const vampAssignments = ts.state_assignments;

    if (!prepLabels || !vampAssignments) return;

    // Compute co-occurrence: for all frames in selected prep state,
    // count how many fall into each VAMP state
    const vampCounts = {};
    let totalInPrep = 0;
    const nFrames = Math.min(prepLabels.length, vampAssignments.length);
    for (let i = 0; i < nFrames; i++) {
        if (prepLabels[i] === state.selectedPrepState) {
            const vs = vampAssignments[i];
            vampCounts[vs] = (vampCounts[vs] || 0) + 1;
            totalInPrep++;
        }
    }

    if (totalInPrep === 0) {
        container.innerHTML = '<div class="alluvial-placeholder">No frames in this prep state.</div>';
        return;
    }

    const stateColors = VISUALIZATION_DATA.config.colors.states;

    // Build targets sorted by fraction
    const targets = Object.entries(vampCounts)
        .map(([vs, count]) => ({ state: parseInt(vs), prob: count / totalInPrep, count: count }))
        .filter(d => d.prob > 0.001)
        .sort((a, b) => b.prob - a.prob);

    const width = container.clientWidth;
    const height = container.clientHeight;

    alluvialSvg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const margin = { top: 30, right: 90, bottom: 20, left: 90 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = alluvialSvg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Title
    alluvialSvg.append('text')
        .attr('x', width / 2)
        .attr('y', 18)
        .attr('text-anchor', 'middle')
        .attr('class', 'alluvial-label')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text(`Prep C${state.selectedPrepState} → VAMP States (Lag ${ts.lagtime})`);

    // Layout
    const sourceX = 0;
    const sourceW = 30;
    const targetX = innerW - 30;
    const targetW = 30;
    const gap = 4;

    // Source node (full height) — prep state color
    g.append('rect')
        .attr('class', 'alluvial-node')
        .attr('x', sourceX)
        .attr('y', 0)
        .attr('width', sourceW)
        .attr('height', innerH)
        .attr('fill', stateColors[state.selectedPrepState % stateColors.length]);

    // Source label
    g.append('text')
        .attr('class', 'alluvial-label')
        .attr('x', sourceX - 6)
        .attr('y', innerH / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '13px')
        .text(`Prep C${state.selectedPrepState}`);

    // Frame count below label
    g.append('text')
        .attr('class', 'alluvial-prob')
        .attr('x', sourceX - 6)
        .attr('y', innerH / 2 + 16)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .text(`(${totalInPrep} frames)`);

    // Compute target positions
    const totalGap = gap * (targets.length - 1);
    const availableH = innerH - totalGap;
    let targetY = 0;
    targets.forEach(t => {
        t.y = targetY;
        t.h = Math.max(8, t.prob * availableH);
        targetY += t.h + gap;
    });

    // Scale if overflow
    if (targetY - gap > innerH) {
        const scale = innerH / (targetY - gap);
        targets.forEach(t => {
            t.y *= scale;
            t.h *= scale;
        });
    }

    // Draw flows and target nodes
    targets.forEach(t => {
        const color = stateColors[t.state % stateColors.length];

        // Flow path (cubic bezier)
        const x0 = sourceX + sourceW;
        const y0_top = (t.y / innerH) * innerH;
        const y0_bot = y0_top + t.h;
        const x1 = targetX;
        const y1_top = t.y;
        const y1_bot = t.y + t.h;
        const cx = (x0 + x1) / 2;

        const path = `M${x0},${y0_top} C${cx},${y0_top} ${cx},${y1_top} ${x1},${y1_top}
                       L${x1},${y1_bot} C${cx},${y1_bot} ${cx},${y0_bot} ${x0},${y0_bot} Z`;

        g.append('path')
            .attr('class', 'alluvial-flow')
            .attr('d', path)
            .attr('fill', color)
            .on('mouseover', function(event) {
                d3.select(this).attr('opacity', 0.8);
                showTooltip(event, `Prep C${state.selectedPrepState} → VAMP S${t.state}<br/>${(t.prob * 100).toFixed(1)}% (${t.count} frames)`);
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', null);
                hideTooltip();
            })
            .on('click', function() {
                onAlluvialTargetClick(t.state);
            });

        // Target node
        g.append('rect')
            .attr('class', 'alluvial-node')
            .attr('x', targetX)
            .attr('y', t.y)
            .attr('width', targetW)
            .attr('height', t.h)
            .attr('fill', color)
            .style('cursor', 'pointer')
            .on('click', function() {
                onAlluvialTargetClick(t.state);
            });

        // Target label
        if (t.h > 14) {
            g.append('text')
                .attr('class', 'alluvial-label')
                .attr('x', targetX + targetW + 6)
                .attr('y', t.y + t.h / 2)
                .attr('dominant-baseline', 'middle')
                .text(`VAMP S${t.state}`);
        }

        // Probability label
        g.append('text')
            .attr('class', 'alluvial-prob')
            .attr('x', targetX + targetW + 6)
            .attr('y', t.y + t.h / 2 + (t.h > 14 ? 14 : 0))
            .attr('dominant-baseline', 'middle')
            .text(`${(t.prob * 100).toFixed(1)}%`);
    });
}

function onAlluvialTargetClick(targetState) {
    // Show protein structure for the clicked target state
    state.selectedVampState = targetState;
    state.selectedFrameIndex = null; // clear frame selection so state structure shows
    embeddingG.selectAll('.embedding-point.selected')
        .classed('selected', false)
        .attr('r', 4);
    updateProteinViewer();
}

// =============================================================================
// Protein viewer
// =============================================================================

function initProteinViewer() {
    const container = document.getElementById('protein-viewer');
    if (!container) return;

    proteinViewer = $3Dmol.createViewer(container, {
        backgroundColor: VISUALIZATION_DATA.config.theme === 'dark' ? '#2d2d2d' : '#f5f5f5'
    });

    if (VISUALIZATION_DATA.protein_structure) {
        proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        const representation = VISUALIZATION_DATA.config.protein.representation;
        proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
        proteinViewer.zoomTo();
        proteinViewer.render();
    }
}

function updateProteinViewer() {
    if (!proteinViewer) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const representation = VISUALIZATION_DATA.config.protein.representation;

    // Priority 1: Selected frame → show its VAMP state's average attention
    if (state.selectedVampState !== null && state.showAttention) {
        proteinViewer.removeAllModels();
        if (VISUALIZATION_DATA.protein_structure) {
            proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        }

        if (ts.state_attention_avg && state.selectedVampState < ts.state_attention_avg.length) {
            const attention = ts.state_attention_avg[state.selectedVampState];
            const colorScale = d3.scaleLinear()
                .domain([0, 1])
                .range([
                    VISUALIZATION_DATA.config.colors.attention.low,
                    VISUALIZATION_DATA.config.colors.attention.high
                ]);

            proteinViewer.setStyle({}, {});
            attention.forEach((value, residueIndex) => {
                const color = colorScale(value);
                proteinViewer.setStyle(
                    { resi: residueIndex + 1 },
                    { [representation]: { color: color } }
                );
            });
        }

        proteinViewer.zoomTo();
        proteinViewer.render();
        return;
    }

    // Priority 2: State structure view (from alluvial click or attention panel)
    const viewState = state.selectedVampState;
    if (viewState !== null && ts.state_structures) {
        const stateData = ts.state_structures[viewState];
        if (stateData && stateData.average) {
            proteinViewer.removeAllModels();

            proteinViewer.addModel(stateData.average, 'pdb');
            proteinViewer.setStyle({model: 0}, {
                [representation]: {
                    colorscheme: {prop: 'b', gradient: 'rwb', min: 0, max: 90}
                }
            });

            if (stateData.representatives) {
                stateData.representatives.forEach((pdb, i) => {
                    proteinViewer.addModel(pdb, 'pdb');
                    proteinViewer.setStyle({model: i + 1}, {
                        [representation]: {opacity: 0.3, color: 'grey'}
                    });
                });
            }

            proteinViewer.zoomTo();
            proteinViewer.render();
            return;
        }
    }

    // Priority 3: Default spectrum coloring
    proteinViewer.removeAllModels();
    if (VISUALIZATION_DATA.protein_structure) {
        proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
        proteinViewer.zoomTo();
    }
    proteinViewer.render();
}

// =============================================================================
// Attention panel (VAMP state chips + 3Dmol viewer)
// =============================================================================

function initAttentionPanel() {
    // Attention viewer is created lazily on first use to avoid
    // blocking init with a second WebGL context.
}

function ensureAttentionViewer() {
    if (attentionViewer) return true;
    const container = document.getElementById('attention-viewer');
    if (!container) return false;
    try {
        attentionViewer = $3Dmol.createViewer(container, {
            backgroundColor: VISUALIZATION_DATA.config.theme === 'dark' ? '#2d2d2d' : '#f5f5f5'
        });
        if (VISUALIZATION_DATA.protein_structure) {
            attentionViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
            const representation = VISUALIZATION_DATA.config.protein.representation;
            attentionViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
            attentionViewer.zoomTo();
            attentionViewer.render();
        }
        return true;
    } catch (e) {
        console.error('Attention viewer init failed:', e);
        return false;
    }
}

function updateAttentionPanel() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const selectorContainer = document.getElementById('attention-state-selector');
    if (!selectorContainer) return;

    selectorContainer.innerHTML = '';

    for (let i = 0; i < ts.n_states; i++) {
        const chip = document.createElement('span');
        chip.className = 'state-chip';
        if (state.selectedAttentionState === i) chip.classList.add('active');
        chip.style.backgroundColor = stateColors[i % stateColors.length];
        chip.textContent = `S${i}`;
        chip.addEventListener('click', () => selectAttentionState(i));
        selectorContainer.appendChild(chip);
    }

    // Clamp if new timescale has fewer states
    if (state.selectedAttentionState !== null && state.selectedAttentionState >= ts.n_states) {
        state.selectedAttentionState = null;
    }

    updateAttentionViewer();
}

function selectAttentionState(stateIndex) {
    if (state.selectedAttentionState === stateIndex) {
        state.selectedAttentionState = null;
    } else {
        state.selectedAttentionState = stateIndex;
    }

    // Update chip highlights
    document.querySelectorAll('#attention-state-selector .state-chip').forEach((chip, i) => {
        chip.classList.toggle('active', i === state.selectedAttentionState);
    });

    updateAttentionViewer();
}

function updateAttentionViewer() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const representation = VISUALIZATION_DATA.config.protein.representation;

    if (state.selectedAttentionState !== null && ts.state_attention_avg) {
        // Lazily create viewer only when a state is actually selected
        if (!ensureAttentionViewer()) return;
        const attnAvg = ts.state_attention_avg[state.selectedAttentionState];
        if (attnAvg) {
            attentionViewer.removeAllModels();
            if (VISUALIZATION_DATA.protein_structure) {
                attentionViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
            }

            const colorScale = d3.scaleLinear()
                .domain([0, 1])
                .range([
                    VISUALIZATION_DATA.config.colors.attention.low,
                    VISUALIZATION_DATA.config.colors.attention.high
                ]);

            attentionViewer.setStyle({}, {});
            attnAvg.forEach((value, residueIndex) => {
                const color = colorScale(value);
                attentionViewer.setStyle(
                    { resi: residueIndex + 1 },
                    { [representation]: { color: color } }
                );
            });

            attentionViewer.zoomTo();
            attentionViewer.render();
            return;
        }
    }

    // Default: spectrum coloring (only if viewer already exists)
    if (!attentionViewer) return;
    attentionViewer.removeAllModels();
    if (VISUALIZATION_DATA.protein_structure) {
        attentionViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        attentionViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
        attentionViewer.zoomTo();
    }
    attentionViewer.render();
}

// =============================================================================
// Transition matrix
// =============================================================================

function initTransitionMatrix() {
    const container = document.getElementById('matrix-container');
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    matrixSvg = d3.select('#matrix-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
}

function updateTransitionMatrix(timescale) {
    matrixSvg.selectAll('*').remove();

    const matrix = timescale.transition_matrix;
    const n = matrix.length;

    const container = document.getElementById('matrix-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    const margin = { top: 40, right: 20, bottom: 20, left: 40 };
    const cellSize = Math.min(
        (width - margin.left - margin.right) / n,
        (height - margin.top - margin.bottom) / n
    );

    const g = matrixSvg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([0, d3.max(matrix.flat())]);

    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const value = matrix[i][j];

            g.append('rect')
                .attr('class', 'matrix-cell')
                .attr('x', j * cellSize)
                .attr('y', i * cellSize)
                .attr('width', cellSize)
                .attr('height', cellSize)
                .attr('fill', colorScale(value))
                .on('mouseover', function(event) {
                    showTooltip(event, `State ${i} → ${j}<br/>P = ${value.toFixed(3)}`);
                })
                .on('mouseout', hideTooltip);

            if (value > 0.001) {
                const fontSize = Math.max(8, Math.min(12, cellSize * 0.35));
                const maxVal = d3.max(matrix.flat());
                const normalizedValue = maxVal > 0 ? value / maxVal : 0;
                g.append('text')
                    .attr('class', 'matrix-value')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .attr('fill', normalizedValue > 0.55 ? '#fff' : '#000')
                    .style('font-size', `${fontSize}px`)
                    .style('pointer-events', 'none')
                    .text(value < 0.1 ? value.toFixed(2) : value.toFixed(1));
            }
        }
    }

    // Row labels
    for (let i = 0; i < n; i++) {
        g.append('text')
            .attr('class', 'matrix-label')
            .attr('x', -5)
            .attr('y', i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .text(`S${i}`);
    }

    // Column labels
    for (let j = 0; j < n; j++) {
        g.append('text')
            .attr('class', 'matrix-label')
            .attr('x', j * cellSize + cellSize / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'bottom')
            .text(`S${j}`);
    }

    // Title
    matrixSvg.append('text')
        .attr('x', width / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('class', 'matrix-label')
        .style('font-size', '14px')
        .style('font-weight', '600')
        .text(`Transition Matrix (Lag ${timescale.lagtime})`);
}

// =============================================================================
// State legend
// =============================================================================

function updateStateLegend(timescale) {
    const legend = document.getElementById('state-legend');
    if (!legend) return;

    const stateColors = VISUALIZATION_DATA.config.colors.states;

    // Use prep cluster labels if available, else VAMP states
    let labels, labelPrefix, n_states;
    if (hasPrep) {
        labels = VISUALIZATION_DATA.prep.cluster_labels;
        labelPrefix = 'Cluster';
        n_states = VISUALIZATION_DATA.prep.n_states;
    } else {
        labels = timescale.state_assignments;
        labelPrefix = 'State';
        n_states = timescale.n_states;
    }

    const counts = {};
    labels.forEach(s => {
        counts[s] = (counts[s] || 0) + 1;
    });

    let html = `<h4>${hasPrep ? 'Prep Clusters' : 'States'}</h4>`;
    for (let i = 0; i < n_states; i++) {
        const color = stateColors[i % stateColors.length];
        const count = counts[i] || 0;
        html += `
            <div class="legend-item">
                <div class="legend-color" style="background-color: ${color}"></div>
                <span class="legend-label">${labelPrefix} ${i}</span>
                <span class="legend-count">${count} frames</span>
            </div>
        `;
    }

    legend.innerHTML = html;
}

// =============================================================================
// Timescale loading
// =============================================================================

function loadTimescale(index) {
    state.currentTimescaleIndex = index;
    const ts = VISUALIZATION_DATA.timescales[index];

    console.log(`Loading timescale ${index}: lagtime ${ts.lagtime}`);

    // Update sidebar button highlighting
    document.querySelectorAll('.timescale-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === index);
    });

    // Recompute VAMP state for selected frame (if any)
    if (state.selectedFrameIndex !== null) {
        state.selectedVampState = (state.selectedFrameIndex < ts.state_assignments.length)
            ? ts.state_assignments[state.selectedFrameIndex] : null;
    }

    // Update timescale-dependent panels
    updateAlluvialPlot();
    updateTransitionMatrix(ts);
    updateStateLegend(ts);
    updateAttentionPanel();
    updateDiagnosticsPanel(ts);
    updateProteinViewer();
}

// =============================================================================
// Tooltip
// =============================================================================

function showTooltip(event, html) {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;

    tooltip.innerHTML = html;
    tooltip.style.left = (event.pageX + 10) + 'px';
    tooltip.style.top = (event.pageY + 10) + 'px';
    tooltip.classList.add('visible');
}

function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
    }
}

// =============================================================================
// Window resize
// =============================================================================

function onWindowResize() {
    // Rebuild embedding plot
    if (embeddingSvg) {
        embeddingSvg.remove();
        embeddingSvg = null;
        embeddingG = null;
        initEmbeddingPlot();
    }

    // Rebuild matrix
    if (matrixSvg) {
        matrixSvg.remove();
        matrixSvg = null;
        initTransitionMatrix();
        const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
        updateTransitionMatrix(ts);
    }

    // Rebuild alluvial
    updateAlluvialPlot();

    // Resize 3Dmol viewers
    if (proteinViewer) proteinViewer.resize();
    if (attentionViewer) attentionViewer.resize();
}

// =============================================================================
// Event listeners
// =============================================================================

function initEventListeners() {
    // Attention toggle
    const attentionToggle = document.getElementById('attention-toggle');
    if (attentionToggle) {
        attentionToggle.checked = state.showAttention;
        attentionToggle.addEventListener('change', (e) => {
            state.showAttention = e.target.checked;
            updateProteinViewer();
        });
    }

    // Protein representation selector
    const representationSelect = document.getElementById('protein-representation');
    if (representationSelect) {
        representationSelect.value = VISUALIZATION_DATA.config.protein.representation;
        representationSelect.addEventListener('change', (e) => {
            VISUALIZATION_DATA.config.protein.representation = e.target.value;
            updateProteinViewer();
            updateAttentionViewer();
        });
    }
}

// =============================================================================
// Diagnostics panel
// =============================================================================

function initDiagnosticsPanel() {
    const hasDiagnostics = VISUALIZATION_DATA.timescales.some(
        ts => ts.metadata && ts.metadata.diagnostics
    );

    const section = document.getElementById('diagnostics-section');
    if (section) {
        section.style.display = hasDiagnostics ? 'block' : 'none';
    }

    // Wire up tab clicks
    document.querySelectorAll('.diag-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.diag-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.diag-tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById('tab-' + tab.dataset.tab);
            if (target) target.classList.add('active');
        });
    });

    // Close on backdrop click
    const modal = document.getElementById('diagnostics-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeDiagnostics();
        });
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDiagnostics();
    });
}

function updateDiagnosticsPanel(timescale) {
    const section = document.getElementById('diagnostics-section');
    const diag = timescale.metadata && timescale.metadata.diagnostics;

    if (!diag || !diag.report) {
        if (section) section.style.display = 'none';
        return;
    }

    if (section) section.style.display = 'block';
    const report = diag.report.diagnostics || {};

    const badge = document.getElementById('diag-recommendation-badge');
    if (badge) {
        const rec = report.recommendation || 'keep';
        badge.textContent = rec;
        badge.className = 'diag-badge ' + rec;
    }

    const quickInfo = document.getElementById('diag-quick-info');
    if (quickInfo) {
        const lines = [];
        lines.push(`Effective states: ${report.effective_n_states || '?'} / ${report.original_n_states || '?'}`);
        lines.push(`Confidence: ${report.confidence || '?'}`);
        if (report.underpopulated_states && report.underpopulated_states.length > 0) {
            lines.push(`Underpopulated: S${report.underpopulated_states.join(', S')}`);
        }
        if (report.merge_groups && report.merge_groups.length > 0) {
            const groups = report.merge_groups.map(g => '{' + g.join(',') + '}').join(' ');
            lines.push(`Redundant groups: ${groups}`);
        }
        quickInfo.innerHTML = lines.join('<br/>');
    }
}

function openDiagnostics() {
    const modal = document.getElementById('diagnostics-modal');
    if (!modal) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const diag = ts.metadata && ts.metadata.diagnostics;
    if (!diag) return;

    const report = diag.report ? diag.report.diagnostics : {};
    const plots = diag.plots || {};

    const title = document.getElementById('diagnostics-title');
    if (title) title.textContent = `State Diagnostics — Lagtime ${ts.lagtime}`;

    const banner = document.getElementById('diagnostics-banner');
    if (banner) {
        const rec = report.recommendation || 'keep';
        banner.className = 'diagnostics-banner ' + rec;
        let bannerText = '';
        if (rec === 'keep') {
            bannerText = `All ${report.original_n_states} states are well-separated. No reduction needed.`;
        } else if (rec === 'retrain') {
            bannerText = `State reduction detected (${report.original_n_states} → ${report.effective_n_states}). Retraining recommended.`;
        }
        banner.textContent = bannerText;
    }

    const plotMap = {
        'summary': 'diagnostic_summary',
        'eigenvalues': 'eigenvalue_spectrum',
        'jsd': 'jsd_heatmap',
        'its': 'implied_timescales',
        'ck': 'ck_test',
    };

    for (const [tabId, plotKey] of Object.entries(plotMap)) {
        const img = document.getElementById(`diag-img-${tabId}`);
        const tab = document.querySelector(`.diag-tab[data-tab="${tabId}"]`);
        if (plots[plotKey]) {
            if (img) { img.src = plots[plotKey]; img.style.display = 'block'; }
            if (tab) tab.classList.remove('hidden');
        } else {
            if (img) img.style.display = 'none';
            if (tab) tab.classList.add('hidden');
        }
    }

    const details = document.getElementById('diag-report-details');
    if (details && report) {
        let html = '<table>';
        html += `<tr><th>Metric</th><th>Value</th></tr>`;
        html += `<tr><td>Original states</td><td>${report.original_n_states || '?'}</td></tr>`;
        html += `<tr><td>Effective states</td><td>${report.effective_n_states || '?'}</td></tr>`;
        html += `<tr><td>Eigenvalue gap suggestion</td><td>${report.eigenvalue_gap_suggestion || '?'}</td></tr>`;
        html += `<tr><td>Confidence</td><td>${report.confidence || '?'}</td></tr>`;
        html += `<tr><td>Recommendation</td><td>${report.recommendation || '?'}</td></tr>`;
        if (report.populations) {
            html += `<tr><td>Populations</td><td>${report.populations.map(p => p.toFixed(3)).join(', ')}</td></tr>`;
        }
        if (report.underpopulated_states && report.underpopulated_states.length > 0) {
            html += `<tr><td>Underpopulated</td><td>S${report.underpopulated_states.join(', S')}</td></tr>`;
        }
        if (report.merge_groups && report.merge_groups.length > 0) {
            const groups = report.merge_groups.map(g => '{S' + g.join(', S') + '}').join(', ');
            html += `<tr><td>Redundant groups</td><td>${groups}</td></tr>`;
        }
        html += '</table>';
        details.innerHTML = html;
    }

    const firstVisibleTab = document.querySelector('.diag-tab:not(.hidden)');
    if (firstVisibleTab) {
        document.querySelectorAll('.diag-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.diag-tab-content').forEach(c => c.classList.remove('active'));
        firstVisibleTab.classList.add('active');
        const tabContent = document.getElementById('tab-' + firstVisibleTab.dataset.tab);
        if (tabContent) tabContent.classList.add('active');
    }

    modal.style.display = 'flex';
}

function closeDiagnostics() {
    const modal = document.getElementById('diagnostics-modal');
    if (modal) modal.style.display = 'none';
}

// =============================================================================
// Start
// =============================================================================

// Use setTimeout to yield to the browser so the loading spinner paints
// before heavy D3/3Dmol initialization work begins.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(init, 0));
} else {
    setTimeout(init, 0);
}
