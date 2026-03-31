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
    },
    proteinPdb: null,            // PDB string currently shown in protein viewer
    proteinPdbLabel: null,       // descriptive label for filename
    attentionPdb: null,          // PDB string currently shown in attention viewer
    attentionPdbLabel: null,     // descriptive label for filename
    // Tab 2: Attention Analysis
    activeMainTab: 'states',
    attTabInitialized: false,
    attTabSelectedState: null,
    attEdgeMatrixCache: {},      // stateIdx → dense matrix
    // Tab 2: Distance probe
    attDistancePairs: [],        // [{res1idx, res2idx}, ...]
    attProteinViewer: null,      // 3Dmol viewer instance (lazy)
    attProteinInitialized: false,
    // Tab 2: RMSD coloring
    attProbeShowRmsd: false,
    attRmsdCache: {},            // timescaleIndex → [{resiIdx, resi, rmsd}, ...]
    // Tab 2: dirty flag — set when timescale changes while tab 2 is hidden
    attTabDirty: false,
    // Tab 1: dirty flag — set when timescale changes while tab 1 is hidden
    statesTabDirty: false,
};

// Helper: map attention index to PDB residue number (resi).
// If VISUALIZATION_DATA.residue_mapping is set, use it; otherwise fall back to index+1.
function attentionIndexToResi(attentionIndex) {
    const mapping = VISUALIZATION_DATA.residue_mapping;
    if (mapping && attentionIndex < mapping.length) {
        return mapping[attentionIndex];
    }
    return attentionIndex + 1;
}

/**
 * Trigger a file download in the browser.
 */
function downloadPdb(pdbString, filename) {
    const blob = new Blob([pdbString], {type: 'chemical/x-pdb'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function downloadProteinPdb() {
    if (!state.proteinPdb) return;
    const name = state.proteinPdbLabel || 'structure';
    downloadPdb(state.proteinPdb, name + '.pdb');
}

function downloadAttentionPdb() {
    if (!state.attentionPdb) return;
    const name = state.attentionPdbLabel || 'state_structure';
    downloadPdb(state.attentionPdb, name + '.pdb');
}

/**
 * Show or hide a download button based on whether PDB data is available.
 */
function updateDownloadButton(buttonId, pdbString) {
    const btn = document.getElementById(buttonId);
    if (btn) btn.style.display = pdbString ? 'block' : 'none';
}

/**
 * Look up the attention value for a given PDB resi on the current view.
 * Returns a number in [0,1] or null if unavailable.
 */
function getAttentionForResi(resi, viewerType) {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    if (!ts) return null;

    // Build reverse mapping: resi → attention index
    const mapping = VISUALIZATION_DATA.residue_mapping;
    let attentionIndex = null;
    if (mapping) {
        attentionIndex = mapping.indexOf(resi);
        if (attentionIndex === -1) return null;
    } else {
        attentionIndex = resi - 1;
    }

    if (viewerType === 'attention') {
        // Use per-state average attention
        if (state.selectedAttentionState !== null && ts.state_attention_avg) {
            const avg = ts.state_attention_avg[state.selectedAttentionState];
            if (avg && attentionIndex < avg.length) return avg[attentionIndex];
        }
    } else {
        // Protein viewer: use per-frame attention if a frame is selected
        if (state.selectedFrameIndex !== null && ts.attention_normalized &&
            state.selectedFrameIndex < ts.attention_normalized.length) {
            const att = ts.attention_normalized[state.selectedFrameIndex];
            if (att && attentionIndex < att.length) return att[attentionIndex];
        }
        // Fall back to state average if a VAMP state is shown
        if (state.selectedVampState !== null && ts.state_attention_avg) {
            const avg = ts.state_attention_avg[state.selectedVampState];
            if (avg && attentionIndex < avg.length) return avg[attentionIndex];
        }
    }
    return null;
}

/**
 * Register hover-to-inspect on a 3Dmol viewer's model 0.
 * Must be called after every removeAllModels/addModel cycle.
 */
function registerResidueHover(viewer, infoElementId, viewerType) {
    if (!viewer) return;

    // On hover: show residue info
    viewer.setHoverable({model: 0}, true,
        function onHover(atom) {
            const infoEl = document.getElementById(infoElementId);
            if (!infoEl) return;

            const attention = getAttentionForResi(atom.resi, viewerType);
            let html = `<span class="residue-name">${atom.resn} ${atom.resi}</span>`;
            if (attention !== null) {
                html += `<span class="residue-attention">Attention: ${attention.toFixed(3)}</span>`;
            }
            infoEl.innerHTML = html;
            infoEl.style.display = 'block';
        },
        function onUnhover() {
            const infoEl = document.getElementById(infoElementId);
            if (infoEl) infoEl.style.display = 'none';
        }
    );
}

// D3.js embedding plot (prep / Graph2Vec)
let embeddingSvg, embeddingG, xScale, yScale, embeddingZoom;
const embeddingMargin = { top: 20, right: 20, bottom: 40, left: 50 };

// D3.js VAMP embedding plot (same coords, VAMP state coloring)
let vampEmbeddingSvg, vampEmbeddingG, vampXScale, vampYScale, vampEmbeddingZoom;

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
        initMainTabs();
        initTimescaleControls();
        initEmbeddingPlot();
        initVampEmbeddingPlot();
        initAlluvialPlot();
        initProteinViewer();
        initTransitionMatrix();
        initDiagnosticsPanel();
        initEventListeners();

        // Load first visible timescale (skips superseded ones)
        loadTimescale(state._firstVisibleIndex || 0);

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

    // Check if any timescales are superseded (not final)
    const hasSuperseded = VISUALIZATION_DATA.timescales.some(
        ts => ts.metadata && ts.metadata.is_final === false
    );

    // Find the first final timescale to auto-select
    let firstVisibleIndex = 0;
    for (let i = 0; i < VISUALIZATION_DATA.timescales.length; i++) {
        const ts = VISUALIZATION_DATA.timescales[i];
        const isFinal = !ts.metadata || ts.metadata.is_final !== false;
        if (isFinal) { firstVisibleIndex = i; break; }
    }

    VISUALIZATION_DATA.timescales.forEach((ts, index) => {
        const isFinal = !ts.metadata || ts.metadata.is_final !== false;

        const btn = document.createElement('button');
        btn.className = 'timescale-btn';
        btn.dataset.index = index;
        if (!isFinal) {
            btn.classList.add('superseded');
            btn.style.display = 'none';
        }
        if (index === firstVisibleIndex) btn.classList.add('active');

        const label = document.createElement('span');
        label.className = 'timescale-label';
        let labelText = `Lag ${ts.lagtime} (${ts.n_states} states)`;
        if (!isFinal) labelText += ' [superseded]';
        label.textContent = labelText;
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

    // Load the first visible timescale
    state._firstVisibleIndex = firstVisibleIndex;
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

    // Click on background to deselect
    embeddingSvg.on('click', function(event) {
        // Only deselect if clicking the background, not a point
        if (event.target.tagName === 'circle') return;
        if (state.selectedFrameIndex === null && state.selectedPrepState === null) return;

        state.selectedFrameIndex = null;
        state.selectedPrepState = null;
        state.selectedVampState = null;
        embeddingG.selectAll('.embedding-point.selected')
            .classed('selected', false).attr('r', 4);
        if (vampEmbeddingG) {
            vampEmbeddingG.selectAll('.embedding-point.selected')
                .classed('selected', false).attr('r', 4);
        }
        updateAlluvialPlot();
        updateProteinViewer();
    });

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
    // Deselect in both plots
    embeddingG.selectAll('.embedding-point.selected')
        .classed('selected', false).attr('r', 4);
    if (vampEmbeddingG) {
        vampEmbeddingG.selectAll('.embedding-point.selected')
            .classed('selected', false).attr('r', 4);
    }

    // Select in prep plot
    d3.select(element).classed('selected', true).attr('r', 6);

    // Also highlight in VAMP plot
    if (vampEmbeddingG) {
        vampEmbeddingG.selectAll('.embedding-point')
            .filter(p => p.index === d.index)
            .classed('selected', true).attr('r', 6);
    }

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
// VAMP Embedding plot (same coords as prep, colored by VAMP states)
// =============================================================================

function initVampEmbeddingPlot() {
    const container = document.getElementById('vamp-embedding-plot');
    if (!container || !hasPrep) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const embData = getEmbeddingData();
    const bounds = embData.bounds;

    const padX = (bounds.max_x - bounds.min_x) * 0.05;
    const padY = (bounds.max_y - bounds.min_y) * 0.05;

    vampXScale = d3.scaleLinear()
        .domain([bounds.min_x - padX, bounds.max_x + padX])
        .range([embeddingMargin.left, width - embeddingMargin.right]);

    vampYScale = d3.scaleLinear()
        .domain([bounds.min_y - padY, bounds.max_y + padY])
        .range([height - embeddingMargin.bottom, embeddingMargin.top]);

    vampEmbeddingSvg = d3.select('#vamp-embedding-plot')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    vampEmbeddingSvg.append('defs').append('clipPath')
        .attr('id', 'vamp-plot-clip')
        .append('rect')
        .attr('x', embeddingMargin.left)
        .attr('y', embeddingMargin.top)
        .attr('width', width - embeddingMargin.left - embeddingMargin.right)
        .attr('height', height - embeddingMargin.top - embeddingMargin.bottom);

    // Gridlines
    vampEmbeddingSvg.append('g')
        .attr('class', 'grid x-grid')
        .attr('transform', `translate(0,${height - embeddingMargin.bottom})`)
        .call(d3.axisBottom(vampXScale).ticks(8)
            .tickSize(-(height - embeddingMargin.top - embeddingMargin.bottom))
            .tickFormat(''));

    vampEmbeddingSvg.append('g')
        .attr('class', 'grid y-grid')
        .attr('transform', `translate(${embeddingMargin.left},0)`)
        .call(d3.axisLeft(vampYScale).ticks(8)
            .tickSize(-(width - embeddingMargin.left - embeddingMargin.right))
            .tickFormat(''));

    // Axes
    vampEmbeddingSvg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - embeddingMargin.bottom})`)
        .call(d3.axisBottom(vampXScale).ticks(8));

    vampEmbeddingSvg.append('g')
        .attr('class', 'y-axis')
        .attr('transform', `translate(${embeddingMargin.left},0)`)
        .call(d3.axisLeft(vampYScale).ticks(8));

    // Points group
    vampEmbeddingG = vampEmbeddingSvg.append('g')
        .attr('clip-path', 'url(#vamp-plot-clip)');

    // Zoom
    vampEmbeddingZoom = d3.zoom()
        .scaleExtent([0.5, 20])
        .on('zoom', onVampEmbeddingZoom);

    vampEmbeddingSvg.call(vampEmbeddingZoom);

    // Click background to deselect
    vampEmbeddingSvg.on('click', function(event) {
        if (event.target.tagName === 'circle') return;
        if (state.selectedFrameIndex === null) return;

        state.selectedFrameIndex = null;
        state.selectedPrepState = null;
        state.selectedVampState = null;
        vampEmbeddingG.selectAll('.embedding-point.selected')
            .classed('selected', false).attr('r', 4);
        embeddingG.selectAll('.embedding-point.selected')
            .classed('selected', false).attr('r', 4);
        updateAlluvialPlot();
        updateProteinViewer();
    });

    drawVampEmbeddingPoints();
}

function drawVampEmbeddingPoints() {
    if (!vampEmbeddingG || !hasPrep) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const embData = getEmbeddingData();

    const data = embData.embeddings.map((point, i) => ({
        x: point[0],
        y: point[1],
        index: i,
        clusterLabel: embData.labels[i],
        vampState: (i < ts.state_assignments.length) ? ts.state_assignments[i] : 0
    }));

    vampEmbeddingG.selectAll('.embedding-point')
        .data(data, d => d.index)
        .enter()
        .append('circle')
        .attr('class', 'embedding-point')
        .attr('r', 4)
        .attr('cx', d => vampXScale(d.x))
        .attr('cy', d => vampYScale(d.y))
        .attr('fill', d => stateColors[d.vampState % stateColors.length])
        .on('mouseover', function(event, d) {
            if (!d3.select(this).classed('selected')) {
                d3.select(this).attr('r', 6);
            }
            const vampState = d.vampState;
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
            onVampFrameClick(d, this);
        });
}

function onVampFrameClick(d, element) {
    // Deselect in both plots
    vampEmbeddingG.selectAll('.embedding-point.selected')
        .classed('selected', false).attr('r', 4);
    embeddingG.selectAll('.embedding-point.selected')
        .classed('selected', false).attr('r', 4);

    // Select in VAMP plot
    d3.select(element).classed('selected', true).attr('r', 6);

    // Also highlight in prep plot
    embeddingG.selectAll('.embedding-point')
        .filter(p => p.index === d.index)
        .classed('selected', true).attr('r', 6);

    state.selectedFrameIndex = d.index;
    state.selectedPrepState = d.clusterLabel;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    state.selectedVampState = (d.index < ts.state_assignments.length)
        ? ts.state_assignments[d.index] : null;

    updateAlluvialPlot();
    updateProteinViewer();
}

function updateVampEmbeddingColors() {
    if (!vampEmbeddingG || !hasPrep) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const stateColors = VISUALIZATION_DATA.config.colors.states;

    vampEmbeddingG.selectAll('.embedding-point')
        .each(function(d) {
            d.vampState = (d.index < ts.state_assignments.length)
                ? ts.state_assignments[d.index] : 0;
        })
        .attr('fill', d => stateColors[d.vampState % stateColors.length]);
}

function onVampEmbeddingZoom(event) {
    const transform = event.transform;
    const newXScale = transform.rescaleX(vampXScale);
    const newYScale = transform.rescaleY(vampYScale);

    const container = document.getElementById('vamp-embedding-plot');
    const width = container.clientWidth;
    const height = container.clientHeight;

    vampEmbeddingSvg.select('.x-axis')
        .call(d3.axisBottom(newXScale).ticks(8));
    vampEmbeddingSvg.select('.y-axis')
        .call(d3.axisLeft(newYScale).ticks(8));

    vampEmbeddingSvg.select('.x-grid')
        .call(d3.axisBottom(newXScale).ticks(8)
            .tickSize(-(height - embeddingMargin.top - embeddingMargin.bottom))
            .tickFormat(''));
    vampEmbeddingSvg.select('.y-grid')
        .call(d3.axisLeft(newYScale).ticks(8)
            .tickSize(-(width - embeddingMargin.left - embeddingMargin.right))
            .tickFormat(''));

    vampEmbeddingG.selectAll('.embedding-point')
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

    container.innerHTML = '';

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const prepLabels = hasPrep ? VISUALIZATION_DATA.prep.cluster_labels : null;
    const vampAssignments = ts.state_assignments;

    if (!prepLabels || !vampAssignments) {
        container.innerHTML = '<div class="alluvial-placeholder">No prep or VAMP data available.</div>';
        return;
    }

    if (state.selectedPrepState !== null) {
        drawSinglePrepAlluvial(container, ts, prepLabels, vampAssignments);
    } else {
        drawFullAlluvial(container, ts, prepLabels, vampAssignments);
    }
}

// Single prep state → VAMP states (when a point is clicked)
function drawSinglePrepAlluvial(container, ts, prepLabels, vampAssignments) {
    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const nFrames = Math.min(prepLabels.length, vampAssignments.length);
    const prepState = state.selectedPrepState;

    // Count VAMP states for this prep cluster
    const vampCounts = {};
    let totalInPrep = 0;
    for (let i = 0; i < nFrames; i++) {
        if (prepLabels[i] === prepState) {
            const vs = vampAssignments[i];
            vampCounts[vs] = (vampCounts[vs] || 0) + 1;
            totalInPrep++;
        }
    }

    if (totalInPrep === 0) {
        container.innerHTML = '<div class="alluvial-placeholder">No frames in this prep state.</div>';
        return;
    }

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

    const margin = { top: 30, right: 90, bottom: 10, left: 90 };
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
        .text(`Prep C${prepState} → VAMP States (Lag ${ts.lagtime})`);

    const sourceX = 0;
    const sourceW = 30;
    const targetX = innerW - 30;
    const targetW = 30;
    const gap = 4;

    // Source node (full height)
    g.append('rect')
        .attr('class', 'alluvial-node')
        .attr('x', sourceX)
        .attr('y', 0)
        .attr('width', sourceW)
        .attr('height', innerH)
        .attr('fill', stateColors[prepState % stateColors.length]);

    g.append('text')
        .attr('class', 'alluvial-label')
        .attr('x', sourceX - 6)
        .attr('y', innerH / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .style('font-size', '13px')
        .text(`C${prepState}`);

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
    if (targetY - gap > innerH) {
        const scale = innerH / (targetY - gap);
        targets.forEach(t => { t.y *= scale; t.h *= scale; });
    }

    // Draw flows and targets
    targets.forEach(t => {
        const color = stateColors[t.state % stateColors.length];

        const x0 = sourceX + sourceW;
        const y0_top = (t.y / innerH) * innerH;
        const y0_bot = y0_top + t.h;
        const x1 = targetX;
        const cx = (x0 + x1) / 2;

        const path = `M${x0},${y0_top} C${cx},${y0_top} ${cx},${t.y} ${x1},${t.y}
                       L${x1},${t.y + t.h} C${cx},${t.y + t.h} ${cx},${y0_bot} ${x0},${y0_bot} Z`;

        g.append('path')
            .attr('class', 'alluvial-flow')
            .attr('d', path)
            .attr('fill', color)
            .on('mouseover', function(event) {
                d3.select(this).attr('opacity', 0.8);
                showTooltip(event, `Prep C${prepState} → VAMP S${t.state}<br/>${(t.prob * 100).toFixed(1)}% (${t.count} frames)`);
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', null);
                hideTooltip();
            })
            .on('click', function() {
                onAlluvialTargetClick(t.state);
            });

        g.append('rect')
            .attr('class', 'alluvial-node')
            .attr('x', targetX)
            .attr('y', t.y)
            .attr('width', targetW)
            .attr('height', t.h)
            .attr('fill', color)
            .style('cursor', 'pointer')
            .on('click', function() { onAlluvialTargetClick(t.state); });

        if (t.h > 14) {
            g.append('text')
                .attr('class', 'alluvial-label')
                .attr('x', targetX + targetW + 6)
                .attr('y', t.y + t.h / 2)
                .attr('dominant-baseline', 'middle')
                .text(`S${t.state}`);
        }

        g.append('text')
            .attr('class', 'alluvial-prob')
            .attr('x', targetX + targetW + 6)
            .attr('y', t.y + t.h / 2 + (t.h > 14 ? 14 : 0))
            .attr('dominant-baseline', 'middle')
            .text(`${(t.prob * 100).toFixed(1)}%`);
    });
}

// Full alluvial: all prep states → all VAMP states (default view)
function drawFullAlluvial(container, ts, prepLabels, vampAssignments) {
    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const nFrames = Math.min(prepLabels.length, vampAssignments.length);
    const nPrepStates = VISUALIZATION_DATA.prep.n_states;
    const nVampStates = ts.n_states;

    // Build co-occurrence matrix
    const cooc = Array.from({ length: nPrepStates }, () => new Array(nVampStates).fill(0));
    const prepCounts = new Array(nPrepStates).fill(0);
    const vampCounts = new Array(nVampStates).fill(0);

    for (let i = 0; i < nFrames; i++) {
        const p = prepLabels[i];
        const v = vampAssignments[i];
        if (p >= 0 && p < nPrepStates && v >= 0 && v < nVampStates) {
            cooc[p][v]++;
            prepCounts[p]++;
            vampCounts[v]++;
        }
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    alluvialSvg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const margin = { top: 30, right: 90, bottom: 10, left: 90 };
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
        .text(`Prep → VAMP States (Lag ${ts.lagtime})`);

    const sourceX = 0;
    const sourceW = 30;
    const targetX = innerW - 30;
    const targetW = 30;
    const nodeGap = 6;

    // Left column: prep nodes
    const prepAvailH = innerH - nodeGap * (nPrepStates - 1);
    const prepNodes = [];
    let py = 0;
    for (let p = 0; p < nPrepStates; p++) {
        const h = Math.max(4, (prepCounts[p] / nFrames) * prepAvailH);
        prepNodes.push({ state: p, y: py, h: h, count: prepCounts[p] });
        py += h + nodeGap;
    }
    if (py - nodeGap > innerH) {
        const scale = innerH / (py - nodeGap);
        prepNodes.forEach(n => { n.y *= scale; n.h *= scale; });
    }

    // Right column: VAMP nodes
    const vampAvailH = innerH - nodeGap * (nVampStates - 1);
    const vampNodes = [];
    let vy = 0;
    for (let v = 0; v < nVampStates; v++) {
        const h = Math.max(4, (vampCounts[v] / nFrames) * vampAvailH);
        vampNodes.push({ state: v, y: vy, h: h, count: vampCounts[v] });
        vy += h + nodeGap;
    }
    if (vy - nodeGap > innerH) {
        const scale = innerH / (vy - nodeGap);
        vampNodes.forEach(n => { n.y *= scale; n.h *= scale; });
    }

    // Flows
    const prepUsed = new Array(nPrepStates).fill(0);
    const vampUsed = new Array(nVampStates).fill(0);

    const flows = [];
    for (let p = 0; p < nPrepStates; p++) {
        for (let v = 0; v < nVampStates; v++) {
            if (cooc[p][v] > 0) {
                flows.push({ prep: p, vamp: v, count: cooc[p][v] });
            }
        }
    }
    flows.sort((a, b) => b.count - a.count);

    flows.forEach(f => {
        const pn = prepNodes[f.prep];
        const vn = vampNodes[f.vamp];

        const srcH = (pn.count > 0) ? (f.count / pn.count) * pn.h : 0;
        const tgtH = (vn.count > 0) ? (f.count / vn.count) * vn.h : 0;

        const srcY = pn.y + prepUsed[f.prep];
        const tgtY = vn.y + vampUsed[f.vamp];
        prepUsed[f.prep] += srcH;
        vampUsed[f.vamp] += tgtH;

        const x0 = sourceX + sourceW;
        const x1 = targetX;
        const cx = (x0 + x1) / 2;

        const path = `M${x0},${srcY} C${cx},${srcY} ${cx},${tgtY} ${x1},${tgtY}
                       L${x1},${tgtY + tgtH} C${cx},${tgtY + tgtH} ${cx},${srcY + srcH} ${x0},${srcY + srcH} Z`;

        const color = stateColors[f.vamp % stateColors.length];

        g.append('path')
            .attr('class', 'alluvial-flow')
            .attr('d', path)
            .attr('fill', color)
            .attr('opacity', 0.5)
            .on('mouseover', function(event) {
                d3.select(this).attr('opacity', 0.85);
                const pct = pn.count > 0 ? ((f.count / pn.count) * 100).toFixed(1) : '0.0';
                showTooltip(event, `Prep C${f.prep} → VAMP S${f.vamp}<br/>${pct}% of C${f.prep} (${f.count} frames)`);
            })
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 0.5);
                hideTooltip();
            })
            .on('click', function() {
                onAlluvialTargetClick(f.vamp);
            });
    });

    // Draw prep nodes
    prepNodes.forEach(pn => {
        const color = stateColors[pn.state % stateColors.length];

        g.append('rect')
            .attr('class', 'alluvial-node')
            .attr('x', sourceX)
            .attr('y', pn.y)
            .attr('width', sourceW)
            .attr('height', pn.h)
            .attr('fill', color);

        if (pn.h > 12) {
            g.append('text')
                .attr('class', 'alluvial-label')
                .attr('x', sourceX - 6)
                .attr('y', pn.y + pn.h / 2)
                .attr('text-anchor', 'end')
                .attr('dominant-baseline', 'middle')
                .style('font-size', '11px')
                .text(`C${pn.state}`);
        }
    });

    // Draw VAMP nodes
    vampNodes.forEach(vn => {
        const color = stateColors[vn.state % stateColors.length];

        g.append('rect')
            .attr('class', 'alluvial-node')
            .attr('x', targetX)
            .attr('y', vn.y)
            .attr('width', targetW)
            .attr('height', vn.h)
            .attr('fill', color)
            .style('cursor', 'pointer')
            .on('click', function() { onAlluvialTargetClick(vn.state); });

        if (vn.h > 12) {
            g.append('text')
                .attr('class', 'alluvial-label')
                .attr('x', targetX + targetW + 6)
                .attr('y', vn.y + vn.h / 2)
                .attr('dominant-baseline', 'middle')
                .style('font-size', '11px')
                .text(`S${vn.state}`);
        }
    });
}

function onAlluvialTargetClick(targetState) {
    // Show state structure in the VAMP State Structures panel
    state.selectedVampState = targetState;
    state.selectedAttentionState = targetState;

    // Update attention panel chip highlights
    document.querySelectorAll('#attention-state-selector .state-chip').forEach((chip, i) => {
        chip.classList.toggle('active', i === targetState);
    });

    updateAttentionViewer();
}

// =============================================================================
// Per-frame PDB construction from template + coordinates
// =============================================================================

function buildPdbFromCoords(frameIndex) {
    const coords = VISUALIZATION_DATA.frame_coordinates;
    const template = VISUALIZATION_DATA.pdb_template;
    if (!coords || !template || frameIndex < 0 || frameIndex >= coords.length) return null;

    const frameCoords = coords[frameIndex];
    const lines = template.split('\n');
    let atomIdx = 0;
    const result = [];

    for (const line of lines) {
        if ((line.startsWith('ATOM') || line.startsWith('HETATM')) && atomIdx < frameCoords.length) {
            const [x, y, z] = frameCoords[atomIdx];
            // PDB format: columns 31-54 are x(8.3f), y(8.3f), z(8.3f)
            const newLine = line.substring(0, 30)
                + x.toFixed(3).padStart(8)
                + y.toFixed(3).padStart(8)
                + z.toFixed(3).padStart(8)
                + line.substring(54);
            result.push(newLine);
            atomIdx++;
        } else {
            result.push(line);
        }
    }
    return result.join('\n');
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

function setProteinViewerInfo(text) {
    const el = document.getElementById('protein-viewer-info');
    if (!el) return;
    if (text) {
        el.textContent = text;
        el.style.display = 'block';
    } else {
        el.style.display = 'none';
    }
}

function updateProteinViewer() {
    if (!proteinViewer) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const representation = VISUALIZATION_DATA.config.protein.representation;

    // Priority 1: Show selected frame's actual conformation
    if (state.selectedFrameIndex !== null && ts.trajectory_frame_indices &&
        VISUALIZATION_DATA.frame_coordinates) {
        const trajFrame = ts.trajectory_frame_indices[state.selectedFrameIndex];
        const pdbStr = buildPdbFromCoords(trajFrame);
        if (pdbStr) {
            proteinViewer.removeAllModels();
            proteinViewer.addModel(pdbStr, 'pdb');

            // Color by this frame's actual attention values
            if (state.showAttention && ts.attention_normalized &&
                state.selectedFrameIndex < ts.attention_normalized.length) {
                const attention = ts.attention_normalized[state.selectedFrameIndex];
                const colorScale = d3.scaleLinear()
                    .domain([0, 0.5, 1])
                    .range(['#0000FF', '#FFFFFF', '#FF0000']);
                proteinViewer.setStyle({}, {});
                attention.forEach((value, residueIndex) => {
                    const color = colorScale(value);
                    proteinViewer.setStyle(
                        { resi: attentionIndexToResi(residueIndex) },
                        { [representation]: { color: color } }
                    );
                });
            } else {
                proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
            }

            // Build info label
            let info = `Frame ${state.selectedFrameIndex}`;
            if (state.selectedPrepState !== null) info += ` · Prep C${state.selectedPrepState}`;
            if (state.selectedVampState !== null) info += ` · VAMP S${state.selectedVampState}`;
            setProteinViewerInfo(info);

            // Track PDB for download
            state.proteinPdb = pdbStr;
            state.proteinPdbLabel = `frame_${state.selectedFrameIndex}`;
            updateDownloadButton('protein-download-btn', state.proteinPdb);

            registerResidueHover(proteinViewer, 'protein-residue-info', 'protein');
            proteinViewer.zoomTo();
            proteinViewer.render();
            return;
        }
    }

    // No per-frame structure shown — clear the info label
    setProteinViewerInfo(null);

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

            // Track PDB for download
            state.proteinPdb = stateData.average;
            state.proteinPdbLabel = `vamp_state_${viewState}_avg`;
            updateDownloadButton('protein-download-btn', state.proteinPdb);

            registerResidueHover(proteinViewer, 'protein-residue-info', 'protein');
            proteinViewer.zoomTo();
            proteinViewer.render();
            return;
        }
    }

    // Priority 4: Default spectrum coloring
    proteinViewer.removeAllModels();
    if (VISUALIZATION_DATA.protein_structure) {
        proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
        proteinViewer.zoomTo();

        // Track PDB for download
        state.proteinPdb = VISUALIZATION_DATA.protein_structure;
        state.proteinPdbLabel = 'default_structure';
        updateDownloadButton('protein-download-btn', state.proteinPdb);
    } else {
        state.proteinPdb = null;
        state.proteinPdbLabel = null;
        updateDownloadButton('protein-download-btn', null);
    }
    registerResidueHover(proteinViewer, 'protein-residue-info', 'protein');
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

    if (state.selectedAttentionState !== null) {
        if (!ensureAttentionViewer()) return;

        attentionViewer.removeAllModels();

        const stateData = ts.state_structures
            ? ts.state_structures[state.selectedAttentionState] : null;

        if (stateData && stateData.average) {
            // Use the pipeline's pre-computed average structure (attention in B-factor)
            attentionViewer.addModel(stateData.average, 'pdb');
            attentionViewer.setStyle({model: 0}, {
                [representation]: {
                    colorscheme: {prop: 'b', gradient: 'rwb', min: 0, max: 90}
                }
            });

            // Overlay representative structures as transparent ghosts
            if (stateData.representatives) {
                stateData.representatives.forEach((pdb, i) => {
                    attentionViewer.addModel(pdb, 'pdb');
                    attentionViewer.setStyle({model: i + 1}, {
                        [representation]: {opacity: 0.5, color: 'grey'}
                    });
                });
            }

            // Track main (average) PDB for download
            state.attentionPdb = stateData.average;
            state.attentionPdbLabel = `vamp_state_${state.selectedAttentionState}_avg`;
        } else if (VISUALIZATION_DATA.protein_structure && ts.state_attention_avg) {
            // Fallback: color global structure by state_attention_avg
            attentionViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
            const attnAvg = ts.state_attention_avg[state.selectedAttentionState];
            if (attnAvg) {
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
                        { resi: attentionIndexToResi(residueIndex) },
                        { [representation]: { color: color } }
                    );
                });
            }

            // Track fallback PDB for download
            state.attentionPdb = VISUALIZATION_DATA.protein_structure;
            state.attentionPdbLabel = `vamp_state_${state.selectedAttentionState}_fallback`;
        }

        updateDownloadButton('attention-download-btn', state.attentionPdb);
        registerResidueHover(attentionViewer, 'attention-residue-info', 'attention');
        attentionViewer.zoomTo();
        attentionViewer.render();
        return;
    }

    // No state selected: clear the viewer and download state
    state.attentionPdb = null;
    state.attentionPdbLabel = null;
    updateDownloadButton('attention-download-btn', null);
    if (!attentionViewer) return;
    attentionViewer.removeAllModels();
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

function refreshAttTab() {
    state.attEdgeMatrixCache = {};
    state.attRmsdCache = {};
    updateAttResidueHeatmap();
    updateAttTabChips();
    updateAttTab();
    updateAttDistanceTable();
    updateAttProteinViewer();
}

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

    // Update timescale-dependent panels.
    // SVG/canvas panels that depend on container dimensions are only updated
    // when their tab is visible; otherwise they are marked dirty.
    updateStateLegend(ts);
    updateAttentionPanel();
    updateDiagnosticsPanel(ts);

    if (state.activeMainTab === 'states') {
        updateVampEmbeddingColors();
        updateAlluvialPlot();
        updateTransitionMatrix(ts);
        updateProteinViewer();
    } else {
        state.statesTabDirty = true;
    }

    // If tab 2 is initialized, refresh it — but only if it's currently visible.
    // Hidden elements have zero dimensions, so D3/canvas renders would break.
    if (state.attTabInitialized) {
        state.attEdgeMatrixCache = {};
        // Reset selected state if new timescale has different state count
        if (state.attTabSelectedState !== null && state.attTabSelectedState >= ts.n_states) {
            state.attTabSelectedState = null;
        }

        if (state.activeMainTab === 'attention') {
            refreshAttTab();
        } else {
            state.attTabDirty = true;
        }
    }
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

    // Rebuild VAMP embedding plot
    if (vampEmbeddingSvg) {
        vampEmbeddingSvg.remove();
        vampEmbeddingSvg = null;
        vampEmbeddingG = null;
        initVampEmbeddingPlot();
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
    if (state.attProteinViewer) state.attProteinViewer.resize();
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

    // Show superseded models toggle
    const hasSuperseded = VISUALIZATION_DATA.timescales.some(
        ts => ts.metadata && ts.metadata.is_final === false
    );

    const showOrigGroup = document.getElementById('show-original-group');
    const showOrigToggle = document.getElementById('show-original-toggle');
    if (showOrigGroup && showOrigToggle && hasSuperseded) {
        showOrigGroup.style.display = '';
        showOrigToggle.addEventListener('change', (e) => {
            const show = e.target.checked;
            document.querySelectorAll('.timescale-btn.superseded').forEach(btn => {
                btn.style.display = show ? '' : 'none';
            });
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
// Main tab switching
// =============================================================================

function initMainTabs() {
    // Hide attention tab if no timescale has edge attention data
    const hasAnyEdgeAttention = VISUALIZATION_DATA.timescales.some(
        ts => ts.state_edge_attention && Object.keys(ts.state_edge_attention).length > 0
    );
    const attTabBtn = document.getElementById('main-tab-attention');
    if (!hasAnyEdgeAttention && attTabBtn) {
        attTabBtn.style.display = 'none';
    }

    document.querySelectorAll('.main-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            if (tabName === state.activeMainTab) return;

            state.activeMainTab = tabName;

            // Update button active states
            document.querySelectorAll('.main-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle tab content visibility
            document.getElementById('tab-content-states').style.display =
                tabName === 'states' ? 'grid' : 'none';
            document.getElementById('tab-content-attention').style.display =
                tabName === 'attention' ? 'grid' : 'none';

            // Lazy-init tab 2 on first visit
            if (tabName === 'attention' && !state.attTabInitialized) {
                initAttTab();
            }

            // Refresh tab 2 if timescale changed while it was hidden
            if (tabName === 'attention' && state.attTabInitialized && state.attTabDirty) {
                state.attTabDirty = false;
                refreshAttTab();
            }

            // Refresh tab 1 if timescale changed while it was hidden
            if (tabName === 'states') {
                if (state.statesTabDirty) {
                    state.statesTabDirty = false;
                    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
                    updateVampEmbeddingColors();
                    updateAlluvialPlot();
                    updateTransitionMatrix(ts);
                    updateProteinViewer();
                }
                if (proteinViewer) proteinViewer.resize();
                if (attentionViewer) attentionViewer.resize();
            }

            // Resize protein distance viewer when switching to tab 2
            if (tabName === 'attention' && state.attProteinViewer) {
                state.attProteinViewer.resize();
            }
        });
    });
}

// =============================================================================
// Tab 2: Attention Analysis
// =============================================================================

function initAttTab() {
    state.attTabInitialized = true;
    state.attEdgeMatrixCache = {};

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const hasData = ts.state_edge_attention && Object.keys(ts.state_edge_attention).length > 0;

    if (!hasData) {
        showAttTabPlaceholder(true);
        return;
    }

    showAttTabPlaceholder(false);
    updateAttTabChips();

    // Wire up residue heatmap top-N slider
    const topnSlider = document.getElementById('att-residue-topn');
    if (topnSlider) {
        topnSlider.addEventListener('input', () => {
            document.getElementById('att-residue-topn-val').textContent = topnSlider.value;
            updateAttResidueHeatmap();
        });
    }

    // Wire up sort-by-residue-number checkbox
    const sortSeqCheckbox = document.getElementById('att-residue-sort-seq');
    if (sortSeqCheckbox) {
        sortSeqCheckbox.addEventListener('change', () => updateAttResidueHeatmap());
    }

    // Wire up edge heatmap threshold slider
    const threshSlider = document.getElementById('att-heatmap-threshold');
    if (threshSlider) {
        threshSlider.addEventListener('input', () => {
            const val = parseFloat(threshSlider.value) / 1000;
            document.getElementById('att-heatmap-threshold-val').textContent = val.toFixed(3);
            updateAttHeatmap();
            updateAttContactsTable();
        });
    }

    // Draw residue heatmap (all states, independent of chip selection)
    updateAttResidueHeatmap();

    // Initialize protein distance probe viewer
    initAttProteinViewer();

    // Auto-select first state for edge heatmap + contacts
    if (ts.state_edge_attention && Object.keys(ts.state_edge_attention).length > 0) {
        state.attTabSelectedState = 0;
        updateAttTabChips();
        updateAttTab();
    }
}

function showAttTabPlaceholder(show) {
    const placeholder = document.getElementById('att-tab-placeholder');
    const panels = ['att-residue-panel', 'att-tab-chips', 'att-heatmap-panel', 'att-contacts-panel', 'att-protein-panel'];
    if (placeholder) placeholder.style.display = show ? 'flex' : 'none';
    panels.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = show ? 'none' : '';
    });
}

function updateAttTabChips() {
    const container = document.getElementById('att-tab-chip-container');
    if (!container) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const stateColors = VISUALIZATION_DATA.config.colors.states;

    container.innerHTML = '';
    for (let i = 0; i < ts.n_states; i++) {
        const chip = document.createElement('span');
        chip.className = 'state-chip';
        if (state.attTabSelectedState === i) chip.classList.add('active');
        chip.style.backgroundColor = stateColors[i % stateColors.length];
        chip.textContent = `S${i}`;
        chip.addEventListener('click', () => {
            state.attTabSelectedState = (state.attTabSelectedState === i) ? null : i;
            state.attEdgeMatrixCache = {};
            updateAttTabChips();
            updateAttTab();
        });
        container.appendChild(chip);
    }
}

function updateAttTab() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const hasData = ts.state_edge_attention && Object.keys(ts.state_edge_attention).length > 0;

    if (!hasData) {
        showAttTabPlaceholder(true);
        return;
    }
    showAttTabPlaceholder(false);

    // Edge heatmap + contacts depend on selected state
    updateAttHeatmap();
    updateAttContactsTable();
    // Update protein distance probe viewer for selected state
    updateAttProteinViewer();
}

// --- Sparse-to-dense reconstruction ---

function getEdgeAttentionMatrix(stateIdx) {
    const cacheKey = `${state.currentTimescaleIndex}_${stateIdx}`;
    if (state.attEdgeMatrixCache[cacheKey]) return state.attEdgeMatrixCache[cacheKey];

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    if (!ts.state_edge_attention) return null;

    const triples = ts.state_edge_attention[stateIdx];
    if (!triples || triples.length === 0) return null;

    // Determine matrix size
    let maxIdx = 0;
    for (const [r, c] of triples) {
        if (r > maxIdx) maxIdx = r;
        if (c > maxIdx) maxIdx = c;
    }
    const n = maxIdx + 1;

    const matrix = Array.from({length: n}, () => new Float32Array(n));
    for (const [r, c, v] of triples) {
        matrix[r][c] = v;
    }

    state.attEdgeMatrixCache[cacheKey] = {matrix, n, triples};
    return state.attEdgeMatrixCache[cacheKey];
}

function getResidueAttentionFromEdges(stateIdx) {
    const data = getEdgeAttentionMatrix(stateIdx);
    if (!data) return null;

    const {matrix, n} = data;
    const sums = new Float32Array(n);

    // Sum columns (incoming attention)
    for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
            sums[c] += matrix[r][c];
        }
    }

    // Min-max normalize
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < n; i++) {
        if (sums[i] < minVal) minVal = sums[i];
        if (sums[i] > maxVal) maxVal = sums[i];
    }
    const range = maxVal - minVal || 1;

    const result = [];
    const names = VISUALIZATION_DATA.residue_names;
    for (let i = 0; i < n; i++) {
        result.push({
            index: i,
            name: (names && i < names.length) ? names[i] : `R${i + 1}`,
            value: sums[i],
            normalized: (sums[i] - minVal) / range
        });
    }

    return result;
}

// --- Normalized residue attention heatmap (all states × residues) ---

function getAllStatesResidueAttention() {
    // Compute per-residue normalized attention for ALL states
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    if (!ts.state_edge_attention) return null;

    const stateResults = [];
    for (let si = 0; si < ts.n_states; si++) {
        const res = getResidueAttentionFromEdges(si);
        if (!res) return null;
        stateResults.push(res);
    }
    return stateResults;
}

function updateAttResidueHeatmap() {
    const container = document.getElementById('att-residue-container');
    if (!container) return;

    const allStates = getAllStatesResidueAttention();
    if (!allStates || allStates.length === 0) {
        container.innerHTML = '';
        return;
    }

    const nStates = allStates.length;
    const nResidues = allStates[0].length;
    const topnSlider = document.getElementById('att-residue-topn');
    const topnLabel = document.getElementById('att-residue-topn-val');
    // Clamp slider max to actual residue count and sync the label
    if (topnSlider) {
        topnSlider.max = nResidues;
        if (parseInt(topnSlider.value) > nResidues) topnSlider.value = nResidues;
    }
    const topN = Math.min(parseInt(topnSlider ? topnSlider.value : 30) || 30, nResidues);
    if (topnLabel) topnLabel.textContent = topN;

    // Find top-N residues by max normalized attention across any state
    const maxPerResidue = allStates[0].map((r, i) => {
        let mx = 0;
        for (let si = 0; si < nStates; si++) {
            if (allStates[si][i].normalized > mx) mx = allStates[si][i].normalized;
        }
        return {index: i, name: r.name, maxNorm: mx};
    });
    maxPerResidue.sort((a, b) => b.maxNorm - a.maxNorm);
    const topResidues = maxPerResidue.slice(0, topN);

    // Re-sort by residue index (sequence order) if checkbox is checked
    const sortBySeq = document.getElementById('att-residue-sort-seq');
    if (sortBySeq && sortBySeq.checked) {
        topResidues.sort((a, b) => a.index - b.index);
    }

    // Build matrix: rows = states, cols = top residues
    const matrix = [];
    for (let si = 0; si < nStates; si++) {
        const row = topResidues.map(r => allStates[si][r.index].normalized);
        matrix.push(row);
    }

    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = {top: 10, right: 60, bottom: 60, left: 70};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const cellW = innerW / topN;
    const cellH = innerH / nStates;

    const svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale: viridis-like
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    // Draw cells
    for (let si = 0; si < nStates; si++) {
        for (let ri = 0; ri < topN; ri++) {
            const val = matrix[si][ri];
            g.append('rect')
                .attr('x', ri * cellW)
                .attr('y', si * cellH)
                .attr('width', cellW)
                .attr('height', cellH)
                .attr('fill', colorScale(val))
                .attr('stroke', 'var(--bg-primary)')
                .attr('stroke-width', 0.5)
                .on('mouseover', function(event) {
                    showTooltip(event,
                        `State ${si} | ${topResidues[ri].name}<br/>Normalized: ${val.toFixed(3)}`);
                })
                .on('mousemove', function(event) {
                    const tooltip = document.getElementById('tooltip');
                    if (tooltip) {
                        tooltip.style.left = (event.pageX + 10) + 'px';
                        tooltip.style.top = (event.pageY + 10) + 'px';
                    }
                })
                .on('mouseout', hideTooltip);
        }
    }

    // Y-axis: state labels
    for (let si = 0; si < nStates; si++) {
        g.append('text')
            .attr('x', -8)
            .attr('y', si * cellH + cellH / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', Math.min(12, cellH * 0.7))
            .attr('font-weight', '600')
            .attr('fill', 'var(--text-secondary)')
            .text(`S${si}`);
    }

    // X-axis: residue labels
    topResidues.forEach((r, i) => {
        g.append('text')
            .attr('x', i * cellW + cellW / 2)
            .attr('y', nStates * cellH + 8)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'hanging')
            .attr('font-size', Math.min(10, cellW * 0.8))
            .attr('fill', 'var(--text-muted)')
            .attr('transform', `rotate(-60, ${i * cellW + cellW / 2}, ${nStates * cellH + 8})`)
            .text(r.name);
    });

    // Color bar
    const barW = 12;
    const barH = innerH;
    const barX = innerW + 10;
    const barSteps = 50;
    for (let i = 0; i < barSteps; i++) {
        const frac = 1 - i / barSteps;
        g.append('rect')
            .attr('x', barX)
            .attr('y', (i / barSteps) * barH)
            .attr('width', barW)
            .attr('height', barH / barSteps + 1)
            .attr('fill', colorScale(frac));
    }
    g.append('text').attr('x', barX + barW + 4).attr('y', 0)
        .attr('font-size', '9px').attr('fill', 'var(--text-muted)')
        .attr('dominant-baseline', 'hanging').text('1.0');
    g.append('text').attr('x', barX + barW + 4).attr('y', barH)
        .attr('font-size', '9px').attr('fill', 'var(--text-muted)')
        .attr('dominant-baseline', 'auto').text('0.0');

    // Axis labels
    svg.append('text')
        .attr('x', margin.left - 40)
        .attr('y', margin.top + innerH / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'auto')
        .attr('transform', `rotate(-90, ${margin.left - 40}, ${margin.top + innerH / 2})`)
        .attr('font-size', '12px')
        .attr('font-weight', '600')
        .attr('fill', 'var(--text-secondary)')
        .text('State');
}

// --- Edge attention heatmap (Canvas-backed) ---

function updateAttHeatmap() {
    const container = document.getElementById('att-heatmap-container');
    if (!container || state.attTabSelectedState === null) {
        if (container) container.innerHTML = '';
        return;
    }

    const data = getEdgeAttentionMatrix(state.attTabSelectedState);
    if (!data) {
        container.innerHTML = '';
        return;
    }

    const {matrix, n} = data;
    const threshold = parseFloat(document.getElementById('att-heatmap-threshold').value) / 1000;

    // Determine which residues have at least one edge above threshold
    const activeResidues = [];
    for (let i = 0; i < n; i++) {
        let hasEdge = false;
        for (let j = 0; j < n; j++) {
            if (matrix[i][j] >= threshold || matrix[j][i] >= threshold) {
                hasEdge = true;
                break;
            }
        }
        if (hasEdge) activeResidues.push(i);
    }

    if (activeResidues.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 40px;">No edges above threshold.</div>';
        return;
    }

    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = {top: 40, right: 10, bottom: 10, left: 40};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const cellSize = Math.min(innerW / activeResidues.length, innerH / activeResidues.length);
    const canvasW = cellSize * activeResidues.length;
    const canvasH = cellSize * activeResidues.length;

    // Find max value for color scale
    let maxVal = 0;
    for (const r of activeResidues) {
        for (const c of activeResidues) {
            if (matrix[r][c] > maxVal) maxVal = matrix[r][c];
        }
    }

    // Zoomable content wrapper
    const zoomGroup = document.createElement('div');
    zoomGroup.style.position = 'absolute';
    zoomGroup.style.top = '0';
    zoomGroup.style.left = '0';
    zoomGroup.style.width = width + 'px';
    zoomGroup.style.height = height + 'px';
    zoomGroup.style.transformOrigin = `${margin.left}px ${margin.top}px`;
    container.appendChild(zoomGroup);

    // Canvas (inside zoomGroup)
    const canvas = document.createElement('canvas');
    canvas.width = Math.ceil(canvasW);
    canvas.height = Math.ceil(canvasH);
    canvas.style.position = 'absolute';
    canvas.style.left = margin.left + 'px';
    canvas.style.top = margin.top + 'px';
    canvas.style.width = canvasW + 'px';
    canvas.style.height = canvasH + 'px';
    canvas.style.imageRendering = 'pixelated';
    zoomGroup.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    const colorFn = d3.scaleSequential(d3.interpolateViridis).domain([0, maxVal || 1]);

    for (let ri = 0; ri < activeResidues.length; ri++) {
        for (let ci = 0; ci < activeResidues.length; ci++) {
            const val = matrix[activeResidues[ri]][activeResidues[ci]];
            ctx.fillStyle = val >= threshold ? colorFn(val) : '#1a1a1a';
            ctx.fillRect(ci * cellSize, ri * cellSize, Math.ceil(cellSize), Math.ceil(cellSize));
        }
    }

    // SVG overlay for axes (inside zoomGroup)
    const names = VISUALIZATION_DATA.residue_names;
    const labelSvg = d3.select(zoomGroup).append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0')
        .style('pointer-events', 'none')
        .style('overflow', 'visible');

    const labelG = labelSvg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Always render labels (they become visible on zoom even if cells are tiny)
    const labelFontSize = Math.max(8, Math.min(10, cellSize * 0.6));
    activeResidues.forEach((resIdx, i) => {
        const label = (names && resIdx < names.length) ? names[resIdx] : `${resIdx + 1}`;
        const shortLabel = label.length > 5 ? label.substring(0, 5) : label;

        // Top labels
        labelG.append('text')
            .attr('class', 'heatmap-label')
            .attr('x', i * cellSize + cellSize / 2)
            .attr('y', -4)
            .attr('text-anchor', 'middle')
            .attr('font-size', labelFontSize)
            .attr('fill', 'var(--text-muted)')
            .text(shortLabel)
            .style('opacity', cellSize >= 12 ? 1 : 0);

        // Left labels
        labelG.append('text')
            .attr('class', 'heatmap-label')
            .attr('x', -4)
            .attr('y', i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', labelFontSize)
            .attr('fill', 'var(--text-muted)')
            .text(shortLabel)
            .style('opacity', cellSize >= 12 ? 1 : 0);
    });

    // Zoom capture layer (on top, covers full container)
    const zoomSvg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0');

    // Transparent rect to capture all pointer events
    zoomSvg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', 'none')
        .attr('pointer-events', 'all');

    // d3.zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([1, 20])
        .translateExtent([[0, 0], [width, height]])
        .on('zoom', (event) => {
            const {x, y, k} = event.transform;
            zoomGroup.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
            // Show labels once zoomed enough that effective cell size >= 12
            const effCell = cellSize * k;
            const labelOpacity = effCell >= 12 ? 1 : 0;
            labelSvg.selectAll('.heatmap-label').style('opacity', labelOpacity);
            // Scale down label font so they don't grow huge
            const scaledFont = Math.max(6, Math.min(10, 10 / k * (effCell >= 12 ? 1 : 0)));
            labelSvg.selectAll('.heatmap-label').attr('font-size', scaledFont);
        });

    zoomSvg.call(zoom);

    // Store references for resetHeatmapZoom
    state._heatmapZoom = zoom;
    state._heatmapZoomSvg = zoomSvg;

    // Tooltip on pointer move (via zoom capture layer)
    zoomSvg.on('mousemove', (event) => {
        // Get the current zoom transform
        const t = d3.zoomTransform(zoomSvg.node());
        // Invert screen coords to data coords
        const containerRect = container.getBoundingClientRect();
        const sx = event.clientX - containerRect.left;
        const sy = event.clientY - containerRect.top;
        // Undo zoom transform, then subtract margin
        const dx = (sx - t.x) / t.k - margin.left;
        const dy = (sy - t.y) / t.k - margin.top;
        const ci = Math.floor(dx / cellSize);
        const ri = Math.floor(dy / cellSize);

        if (ri >= 0 && ri < activeResidues.length && ci >= 0 && ci < activeResidues.length) {
            const srcIdx = activeResidues[ri];
            const tgtIdx = activeResidues[ci];
            const val = matrix[srcIdx][tgtIdx];
            const srcName = (names && srcIdx < names.length) ? names[srcIdx] : `R${srcIdx + 1}`;
            const tgtName = (names && tgtIdx < names.length) ? names[tgtIdx] : `R${tgtIdx + 1}`;
            showTooltip(event, `${srcName} → ${tgtName}<br/>Attention: ${val.toFixed(4)}`);
        } else {
            hideTooltip();
        }
    });
    zoomSvg.on('mouseout', hideTooltip);
}

function resetHeatmapZoom() {
    if (state._heatmapZoom && state._heatmapZoomSvg) {
        state._heatmapZoomSvg.call(state._heatmapZoom.transform, d3.zoomIdentity);
    }
}

// --- Top contacts table ---

function updateAttContactsTable() {
    const container = document.getElementById('att-contacts-container');
    if (!container || state.attTabSelectedState === null) {
        if (container) container.innerHTML = '';
        return;
    }

    const data = getEdgeAttentionMatrix(state.attTabSelectedState);
    if (!data) {
        container.innerHTML = '';
        return;
    }

    const {triples} = data;
    const threshold = parseFloat(document.getElementById('att-heatmap-threshold').value) / 1000;
    const names = VISUALIZATION_DATA.residue_names;

    const filtered = triples
        .filter(([r, c, v]) => v >= threshold)
        .sort((a, b) => b[2] - a[2])
        .slice(0, 100);

    let html = '<table><thead><tr><th>#</th><th>Source</th><th>Target</th><th>Attention</th></tr></thead><tbody>';
    filtered.forEach(([r, c, v], i) => {
        const srcName = (names && r < names.length) ? names[r] : `R${r + 1}`;
        const tgtName = (names && c < names.length) ? names[c] : `R${c + 1}`;
        html += `<tr><td>${i + 1}</td><td>${srcName}</td><td>${tgtName}</td><td>${v.toFixed(4)}</td></tr>`;
    });
    html += '</tbody></table>';

    container.innerHTML = html;
}

// --- Download functions ---

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], {type: mimeType});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function svgToPng(svgElement, filename, scale) {
    scale = scale || 2;
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d');
        ctx.scale(scale, scale);
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        canvas.toBlob(function(blob) {
            const dlUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = dlUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(dlUrl);
        }, 'image/png');
    };
    img.src = url;
}

function downloadAttResiduePng() {
    const svg = document.querySelector('#att-residue-container svg');
    if (svg) svgToPng(svg, 'residue_attention_heatmap.png', 3);
}

function downloadAttResidueCsv() {
    const allStates = getAllStatesResidueAttention();
    if (!allStates) return;
    const nStates = allStates.length;
    const nResidues = allStates[0].length;

    let csv = 'Residue';
    for (let si = 0; si < nStates; si++) csv += `,State${si}_raw,State${si}_normalized`;
    csv += '\n';

    for (let ri = 0; ri < nResidues; ri++) {
        csv += allStates[0][ri].name;
        for (let si = 0; si < nStates; si++) {
            csv += `,${allStates[si][ri].value.toFixed(6)},${allStates[si][ri].normalized.toFixed(6)}`;
        }
        csv += '\n';
    }
    downloadFile(csv, 'residue_attention_all_states.csv', 'text/csv');
}

function downloadAttHeatmapPng() {
    const canvas = document.querySelector('#att-heatmap-container canvas');
    if (!canvas) return;
    canvas.toBlob(function(blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'edge_attention_heatmap.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 'image/png');
}

function downloadAttHeatmapCsv() {
    if (state.attTabSelectedState === null) return;
    const data = getEdgeAttentionMatrix(state.attTabSelectedState);
    if (!data) return;
    const {triples} = data;
    const names = VISUALIZATION_DATA.residue_names;
    let csv = 'Source,Target,Attention\n';
    const sorted = [...triples].sort((a, b) => b[2] - a[2]);
    sorted.forEach(([r, c, v]) => {
        const srcName = (names && r < names.length) ? names[r] : `R${r + 1}`;
        const tgtName = (names && c < names.length) ? names[c] : `R${c + 1}`;
        csv += `${srcName},${tgtName},${v.toFixed(6)}\n`;
    });
    downloadFile(csv, 'edge_attention.csv', 'text/csv');
}

function downloadAttContactsCsv() {
    if (state.attTabSelectedState === null) return;
    const data = getEdgeAttentionMatrix(state.attTabSelectedState);
    if (!data) return;
    const {triples} = data;
    const threshold = parseFloat(document.getElementById('att-heatmap-threshold').value) / 1000;
    const names = VISUALIZATION_DATA.residue_names;
    const filtered = triples.filter(([r, c, v]) => v >= threshold).sort((a, b) => b[2] - a[2]);
    let csv = 'Rank,Source,Target,Attention\n';
    filtered.forEach(([r, c, v], i) => {
        const srcName = (names && r < names.length) ? names[r] : `R${r + 1}`;
        const tgtName = (names && c < names.length) ? names[c] : `R${c + 1}`;
        csv += `${i + 1},${srcName},${tgtName},${v.toFixed(6)}\n`;
    });
    downloadFile(csv, 'top_contacts.csv', 'text/csv');
}

function downloadAlluvialPng() {
    const svgEl = document.querySelector('#alluvial-plot svg');
    if (!svgEl) return;
    svgToPng(svgEl, 'prep_vamp_state_mapping.png', 3);
}

function downloadAlluvialCsv() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    if (!ts) return;
    const prepLabels = VISUALIZATION_DATA.prep ? VISUALIZATION_DATA.prep.labels : null;
    const vampAssignments = ts.state_assignments;
    if (!prepLabels || !vampAssignments) return;

    const nFrames = Math.min(prepLabels.length, vampAssignments.length);
    const nPrepStates = VISUALIZATION_DATA.prep.n_states;
    const nVampStates = ts.n_states;

    // Build co-occurrence matrix
    const cooc = Array.from({length: nPrepStates}, () => new Array(nVampStates).fill(0));
    for (let i = 0; i < nFrames; i++) {
        const p = prepLabels[i];
        const v = vampAssignments[i];
        if (p >= 0 && p < nPrepStates && v >= 0 && v < nVampStates) {
            cooc[p][v]++;
        }
    }

    // Header
    let csv = 'Prep \\ VAMP,' + Array.from({length: nVampStates}, (_, j) => `VAMP ${j}`).join(',') + ',Total\n';
    for (let p = 0; p < nPrepStates; p++) {
        const rowTotal = cooc[p].reduce((a, b) => a + b, 0);
        csv += `Prep ${p},` + cooc[p].join(',') + `,${rowTotal}\n`;
    }
    // Totals row
    csv += 'Total';
    for (let v = 0; v < nVampStates; v++) {
        let colTotal = 0;
        for (let p = 0; p < nPrepStates; p++) colTotal += cooc[p][v];
        csv += `,${colTotal}`;
    }
    csv += `,${nFrames}\n`;

    downloadFile(csv, 'prep_vamp_state_mapping.csv', 'text/csv');
}

function downloadTransitionMatrixPng() {
    const svgEl = document.querySelector('#matrix-container svg');
    if (!svgEl) return;
    svgToPng(svgEl, 'transition_matrix.png', 3);
}

function downloadTransitionMatrixCsv() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    if (!ts || !ts.transition_matrix) return;
    const matrix = ts.transition_matrix;
    const n = matrix.length;
    // Header row
    let csv = ',' + Array.from({length: n}, (_, j) => `State ${j}`).join(',') + '\n';
    for (let i = 0; i < n; i++) {
        csv += `State ${i}`;
        for (let j = 0; j < n; j++) {
            csv += `,${matrix[i][j].toFixed(6)}`;
        }
        csv += '\n';
    }
    downloadFile(csv, 'transition_matrix.csv', 'text/csv');
}

// =============================================================================
// Tab 2: Protein Distance Probe Viewer
// =============================================================================

function initAttProteinViewer() {
    if (state.attProteinInitialized) return;
    state.attProteinInitialized = true;

    // Populate residue dropdowns
    const names = VISUALIZATION_DATA.residue_names;
    const sel1 = document.getElementById('att-pair-res1');
    const sel2 = document.getElementById('att-pair-res2');
    if (!sel1 || !sel2 || !names || names.length === 0) return;

    names.forEach((name, i) => {
        const opt1 = document.createElement('option');
        opt1.value = i;
        opt1.textContent = name;
        sel1.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = i;
        opt2.textContent = name;
        sel2.appendChild(opt2);
    });

    // Default second dropdown to a different residue
    if (names.length > 1) sel2.value = 1;

    // Create 3Dmol viewer
    const container = document.getElementById('att-protein-viewer-container');
    if (container && $3Dmol) {
        state.attProteinViewer = $3Dmol.createViewer(container, {
            backgroundColor: getComputedStyle(document.documentElement)
                .getPropertyValue('--bg-secondary').trim() || '#2d2d2d'
        });
    }

    // RMSD toggle
    const rmsdToggle = document.getElementById('att-rmsd-toggle');
    if (rmsdToggle) {
        rmsdToggle.addEventListener('change', () => {
            state.attProbeShowRmsd = rmsdToggle.checked;
            const legend = document.getElementById('att-rmsd-legend');
            if (legend) legend.style.display = rmsdToggle.checked ? 'flex' : 'none';
            updateAttProteinViewer();
        });
    }

    // Load first state if available
    updateAttProteinViewer();
}

function getCACoordFromPdb(pdbString, resi) {
    // Parse PDB text, find first CA atom matching the given residue sequence number
    const lines = pdbString.split('\n');
    for (const line of lines) {
        if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue;
        const atomName = line.substring(12, 16).trim();
        if (atomName !== 'CA') continue;
        const lineResi = parseInt(line.substring(22, 26).trim(), 10);
        if (lineResi === resi) {
            return {
                x: parseFloat(line.substring(30, 38).trim()),
                y: parseFloat(line.substring(38, 46).trim()),
                z: parseFloat(line.substring(46, 54).trim())
            };
        }
    }
    return null;
}

function computePairDistances() {
    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const names = VISUALIZATION_DATA.residue_names;
    const results = [];

    for (const pair of state.attDistancePairs) {
        const resi1 = attentionIndexToResi(pair.res1idx);
        const resi2 = attentionIndexToResi(pair.res2idx);
        const res1name = (names && pair.res1idx < names.length) ? names[pair.res1idx] : `R${resi1}`;
        const res2name = (names && pair.res2idx < names.length) ? names[pair.res2idx] : `R${resi2}`;

        const distances = [];
        for (let si = 0; si < ts.n_states; si++) {
            const stateData = ts.state_structures ? ts.state_structures[si] : null;
            if (stateData && stateData.average) {
                const c1 = getCACoordFromPdb(stateData.average, resi1);
                const c2 = getCACoordFromPdb(stateData.average, resi2);
                if (c1 && c2) {
                    const dx = c1.x - c2.x, dy = c1.y - c2.y, dz = c1.z - c2.z;
                    distances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
                } else {
                    distances.push(null);
                }
            } else {
                distances.push(null);
            }
        }

        const valid = distances.filter(d => d !== null);
        const delta = valid.length >= 2 ? Math.max(...valid) - Math.min(...valid) : null;

        results.push({res1name, res2name, res1idx: pair.res1idx, res2idx: pair.res2idx, distances, delta});
    }

    return results;
}

function addDistancePair() {
    const sel1 = document.getElementById('att-pair-res1');
    const sel2 = document.getElementById('att-pair-res2');
    if (!sel1 || !sel2) return;

    const res1idx = parseInt(sel1.value, 10);
    const res2idx = parseInt(sel2.value, 10);
    if (res1idx === res2idx) return;

    // Avoid duplicates
    const exists = state.attDistancePairs.some(
        p => (p.res1idx === res1idx && p.res2idx === res2idx) ||
             (p.res1idx === res2idx && p.res2idx === res1idx)
    );
    if (exists) return;

    state.attDistancePairs.push({res1idx, res2idx});
    updateAttDistanceTable();
    updateAttProteinViewer();
}

function removeDistancePair(index) {
    state.attDistancePairs.splice(index, 1);
    updateAttDistanceTable();
    updateAttProteinViewer();
}

function clearDistancePairs() {
    state.attDistancePairs = [];
    updateAttDistanceTable();
    updateAttProteinViewer();
}

function updateAttDistanceTable() {
    const table = document.getElementById('att-distance-table');
    if (!table) return;

    if (state.attDistancePairs.length === 0) {
        table.innerHTML = '';
        return;
    }

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const results = computePairDistances();

    // Build header
    let html = '<thead><tr><th>Pair</th>';
    for (let si = 0; si < ts.n_states; si++) {
        html += `<th>S${si}</th>`;
    }
    html += '<th>&Delta;</th><th></th></tr></thead><tbody>';

    results.forEach((r, i) => {
        html += `<tr><td>${r.res1name} &harr; ${r.res2name}</td>`;
        r.distances.forEach(d => {
            html += `<td>${d !== null ? d.toFixed(1) + ' &Aring;' : '&mdash;'}</td>`;
        });
        html += `<td>${r.delta !== null ? r.delta.toFixed(1) + ' &Aring;' : '&mdash;'}</td>`;
        html += `<td><button class="remove-pair" onclick="removeDistancePair(${i})">&times;</button></td>`;
        html += '</tr>';
    });

    html += '</tbody>';
    table.innerHTML = html;
}

function computePerResidueRmsd() {
    const tsIdx = state.currentTimescaleIndex;
    if (state.attRmsdCache[tsIdx]) return state.attRmsdCache[tsIdx];

    const ts = VISUALIZATION_DATA.timescales[tsIdx];
    if (!ts.state_structures) return [];

    const names = VISUALIZATION_DATA.residue_names;
    const nResidues = names ? names.length : 0;
    if (nResidues === 0) return [];

    // Collect all available state PDBs
    const pdbs = [];
    for (let si = 0; si < ts.n_states; si++) {
        const sd = ts.state_structures[si];
        if (sd && sd.average) pdbs.push(sd.average);
    }
    if (pdbs.length < 2) return [];

    const result = [];
    for (let ri = 0; ri < nResidues; ri++) {
        const resi = attentionIndexToResi(ri);
        const coords = [];
        for (const pdb of pdbs) {
            const c = getCACoordFromPdb(pdb, resi);
            if (c) coords.push(c);
        }
        if (coords.length < 2) {
            result.push({resiIdx: ri, resi, rmsd: 0});
            continue;
        }
        // Centroid
        let cx = 0, cy = 0, cz = 0;
        for (const c of coords) { cx += c.x; cy += c.y; cz += c.z; }
        cx /= coords.length; cy /= coords.length; cz /= coords.length;
        // RMSD from centroid
        let sumSq = 0;
        for (const c of coords) {
            const dx = c.x - cx, dy = c.y - cy, dz = c.z - cz;
            sumSq += dx * dx + dy * dy + dz * dz;
        }
        result.push({resiIdx: ri, resi, rmsd: Math.sqrt(sumSq / coords.length)});
    }

    state.attRmsdCache[tsIdx] = result;
    return result;
}

function updateAttProteinViewer() {
    const viewer = state.attProteinViewer;
    if (!viewer) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const stateIdx = state.attTabSelectedState;
    const placeholder = document.getElementById('att-protein-placeholder');
    const viewerWrap = document.getElementById('att-protein-viewer-wrap');
    const controls = document.getElementById('att-protein-controls');
    const tableWrap = document.getElementById('att-distance-table-wrap');

    // Check if structures are available for any state
    const hasStructures = ts.state_structures && Object.keys(ts.state_structures).length > 0;

    if (!hasStructures) {
        if (placeholder) placeholder.style.display = 'flex';
        if (viewerWrap) viewerWrap.style.display = 'none';
        if (controls) controls.style.display = 'none';
        if (tableWrap) tableWrap.style.display = 'none';
        return;
    }

    if (placeholder) placeholder.style.display = 'none';
    if (viewerWrap) viewerWrap.style.display = '';
    if (controls) controls.style.display = '';
    if (tableWrap) tableWrap.style.display = '';

    viewer.removeAllModels();
    viewer.removeAllShapes();
    viewer.removeAllLabels();

    // Load the selected state structure (or first available)
    const loadIdx = stateIdx !== null ? stateIdx : 0;
    const stateData = ts.state_structures[loadIdx];
    if (stateData && stateData.average) {
        viewer.addModel(stateData.average, 'pdb');

        const rep = VISUALIZATION_DATA.config.protein.representation || 'cartoon';

        if (state.attProbeShowRmsd) {
            const rmsdData = computePerResidueRmsd();
            if (rmsdData.length > 0) {
                const vals = rmsdData.map(r => r.rmsd);
                const minV = Math.min(...vals);
                const maxV = Math.max(...vals);
                const range = maxV - minV || 1;
                const colorScale = d3.scaleLinear()
                    .domain([0, 0.5, 1])
                    .range(['#00CC66', '#FFDD00', '#CC00CC']);
                viewer.setStyle({}, {});
                rmsdData.forEach(r => {
                    const norm = (r.rmsd - minV) / range;
                    const color = colorScale(norm);
                    viewer.setStyle({resi: r.resi}, {[rep]: {color: color}});
                });
            } else {
                viewer.setStyle({}, {[rep]: {color: 'spectrum'}});
            }
        } else {
            viewer.setStyle({}, {[rep]: {color: 'spectrum'}});
        }

        // Draw distance probe lines
        const stateColors = VISUALIZATION_DATA.config.colors.states;
        for (const pair of state.attDistancePairs) {
            const resi1 = attentionIndexToResi(pair.res1idx);
            const resi2 = attentionIndexToResi(pair.res2idx);
            const c1 = getCACoordFromPdb(stateData.average, resi1);
            const c2 = getCACoordFromPdb(stateData.average, resi2);
            if (c1 && c2) {
                viewer.addCylinder({
                    start: c1, end: c2,
                    radius: 0.15,
                    color: '#FFFF00',
                    dashed: true,
                    dashLength: 0.4,
                    gapLength: 0.2,
                    fromCap: 2, toCap: 2
                });
                // Label at midpoint
                const dx = c1.x - c2.x, dy = c1.y - c2.y, dz = c1.z - c2.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                viewer.addLabel(dist.toFixed(1) + ' \u00C5', {
                    position: {x: (c1.x + c2.x) / 2, y: (c1.y + c2.y) / 2, z: (c1.z + c2.z) / 2},
                    fontSize: 12,
                    fontColor: 'white',
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    borderRadius: 4,
                    padding: 2
                });
            }
        }

        viewer.zoomTo();
        viewer.render();
    }

    viewer.resize();
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
