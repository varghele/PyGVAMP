// MD Trajectory Visualization - Main JavaScript
// This file is rendered as a Jinja2 template

// Global state
const state = {
    currentTimescaleIndex: 0,
    selectedFrameIndex: null,
    selectedState: null,
    hoveredFrameIndex: null,
    showAttention: true,
    camera: {
        rotation: true,
        rotationSpeed: 0.5
    },
    protein: {
        representation: 'cartoon',
        colorScheme: 'attention'
    }
};

// Three.js scene components
let embeddingScene, embeddingCamera, embeddingRenderer, embeddingControls;
let pointClouds = [];
let selectedMarker = null;

// 3Dmol.js viewer
let proteinViewer;

// D3.js matrix
let matrixSvg;

// D3.js state transitions
let transitionsSvg;

// Initialize visualization
function init() {
    console.log('Initializing MD Trajectory Visualization...');
    console.log(`Loaded ${VISUALIZATION_DATA.timescales.length} timescales`);

    // Apply theme
    if (VISUALIZATION_DATA.config.theme === 'light') {
        document.body.classList.add('light-theme');
    }

    // Initialize all components
    initTimescaleControls();
    initEmbeddingViewer();
    initProteinViewer();
    initTransitionMatrix();
    initStateDetails();
    initDiagnosticsPanel();
    initEventListeners();

    // Load first timescale
    loadTimescale(0);

    // Hide loading screen
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }

    // Start animation loop
    animate();

    console.log('Initialization complete');
}

// Initialize timescale controls
function initTimescaleControls() {
    const container = document.getElementById('timescale-list');
    container.innerHTML = '';

    VISUALIZATION_DATA.timescales.forEach((ts, index) => {
        const btn = document.createElement('button');
        btn.className = 'timescale-btn';
        if (index === 0) btn.classList.add('active');

        btn.innerHTML = `
            <div>
                <div class="timescale-label">Lagtime ${ts.lagtime}</div>
                <div class="timescale-details">${ts.n_frames} frames, ${ts.n_states} states</div>
            </div>
        `;

        btn.addEventListener('click', () => {
            document.querySelectorAll('.timescale-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTimescale(index);
        });

        container.appendChild(btn);
    });
}

// Initialize Three.js embedding viewer
function initEmbeddingViewer() {
    const container = document.getElementById('embedding-viewer');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    embeddingScene = new THREE.Scene();
    embeddingScene.background = new THREE.Color(
        VISUALIZATION_DATA.config.theme === 'dark' ? 0x1a1a1a : 0xffffff
    );

    // Camera
    embeddingCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    embeddingCamera.position.set(0, 0, 15);

    // Renderer
    embeddingRenderer = new THREE.WebGLRenderer({ antialias: true });
    embeddingRenderer.setSize(width, height);
    embeddingRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(embeddingRenderer.domElement);

    // Controls
    embeddingControls = new THREE.OrbitControls(embeddingCamera, embeddingRenderer.domElement);
    embeddingControls.enableDamping = true;
    embeddingControls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    embeddingScene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(10, 10, 10);
    embeddingScene.add(directionalLight);

    // Grid helper
    const gridSize = 20;
    const gridDivisions = 20;
    const gridColor = VISUALIZATION_DATA.config.theme === 'dark' ? 0x444444 : 0xcccccc;
    const grid = new THREE.GridHelper(gridSize, gridDivisions, gridColor, gridColor);
    embeddingScene.add(grid);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    embeddingScene.add(axesHelper);

    // Raycaster for picking
    window.raycaster = new THREE.Raycaster();
    window.mouse = new THREE.Vector2();

    // Mouse events
    embeddingRenderer.domElement.addEventListener('mousemove', onEmbeddingMouseMove);
    embeddingRenderer.domElement.addEventListener('click', onEmbeddingClick);

    // Resize handler
    window.addEventListener('resize', onWindowResize);
}

// Initialize 3Dmol.js protein viewer
function initProteinViewer() {
    const container = document.getElementById('protein-viewer');

    // Create 3Dmol viewer
    proteinViewer = $3Dmol.createViewer(container, {
        backgroundColor: VISUALIZATION_DATA.config.theme === 'dark' ? '#2d2d2d' : '#f5f5f5'
    });

    // Load protein structure if available
    if (VISUALIZATION_DATA.protein_structure) {
        proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');

        // Set initial representation
        const representation = VISUALIZATION_DATA.config.protein.representation;
        proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });

        proteinViewer.zoomTo();
        proteinViewer.render();
    }
}

// Initialize D3.js transition matrix
function initTransitionMatrix() {
    const container = document.getElementById('matrix-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    matrixSvg = d3.select('#matrix-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
}

// Load timescale data
function loadTimescale(index) {
    state.currentTimescaleIndex = index;
    const ts = VISUALIZATION_DATA.timescales[index];

    console.log(`Loading timescale ${index}: lagtime ${ts.lagtime}`);

    // Clear previous point clouds
    pointClouds.forEach(cloud => embeddingScene.remove(cloud));
    pointClouds = [];

    // Create point clouds for all timescales with this one highlighted
    VISUALIZATION_DATA.timescales.forEach((timescale, i) => {
        const isActive = i === index;
        const zPosition = i * VISUALIZATION_DATA.config.embedding.z_spacing;

        const pointCloud = createPointCloud(timescale, zPosition, isActive);
        embeddingScene.add(pointCloud);
        pointClouds.push(pointCloud);
    });

    // Update transition matrix
    updateTransitionMatrix(ts);

    // Update state legend
    updateStateLegend(ts);

    // Update state details panel
    updateStateDetails(ts);

    // Update diagnostics panel
    updateDiagnosticsPanel(ts);

    // Reset frame selection (keep state selection across timescale switches)
    state.selectedFrameIndex = null;
    updateProteinViewer();
}

// Create point cloud from embeddings
function createPointCloud(timescale, zPosition, isActive) {
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    const bounds = VISUALIZATION_DATA.bounds;
    const scaleX = 10 / (bounds.max_x - bounds.min_x);
    const scaleY = 10 / (bounds.max_y - bounds.min_y);
    const centerX = (bounds.max_x + bounds.min_x) / 2;
    const centerY = (bounds.max_y + bounds.min_y) / 2;

    const stateColors = VISUALIZATION_DATA.config.colors.states;

    timescale.embeddings.forEach((point, i) => {
        const x = (point[0] - centerX) * scaleX;
        const y = (point[1] - centerY) * scaleY;
        const z = zPosition;

        positions.push(x, y, z);

        // Color by state if active, grey if not
        if (isActive) {
            const state = timescale.state_assignments[i];
            const colorHex = stateColors[state % stateColors.length];
            const color = new THREE.Color(colorHex);
            colors.push(color.r, color.g, color.b);
        } else {
            // Grey for non-active layers
            colors.push(0.5, 0.5, 0.5);
        }
    });

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: VISUALIZATION_DATA.config.embedding.point_size,
        vertexColors: true,
        transparent: false,
        opacity: 1.0
    });

    const pointCloud = new THREE.Points(geometry, material);
    pointCloud.userData = {
        timescaleIndex: VISUALIZATION_DATA.timescales.indexOf(timescale),
        isActive: isActive
    };

    return pointCloud;
}

// Update transition matrix
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

            // Add probability text in every cell
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

    // Add row labels
    for (let i = 0; i < n; i++) {
        g.append('text')
            .attr('class', 'matrix-label')
            .attr('x', -5)
            .attr('y', i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .text(`S${i}`);
    }

    // Add column labels
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
        .text(`Transition Matrix (Lagtime ${timescale.lagtime})`);
}

// Update state legend
function updateStateLegend(timescale) {
    const legend = document.getElementById('state-legend');
    if (!legend) return;

    const stateColors = VISUALIZATION_DATA.config.colors.states;
    const stateCounts = {};

    timescale.state_assignments.forEach(state => {
        stateCounts[state] = (stateCounts[state] || 0) + 1;
    });

    let html = '<h4>States</h4>';
    for (let i = 0; i < timescale.n_states; i++) {
        const color = stateColors[i % stateColors.length];
        const count = stateCounts[i] || 0;
        html += `
            <div class="legend-item">
                <div class="legend-color" style="background-color: ${color}"></div>
                <span class="legend-label">State ${i}</span>
                <span class="legend-count">${count} frames</span>
            </div>
        `;
    }

    legend.innerHTML = html;
}

// Update protein viewer with attention
function updateProteinViewer() {
    if (!proteinViewer) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const representation = VISUALIZATION_DATA.config.protein.representation;

    // Priority 1: Single-frame attention coloring
    if (state.selectedFrameIndex !== null && state.showAttention) {
        proteinViewer.removeAllModels();
        if (VISUALIZATION_DATA.protein_structure) {
            proteinViewer.addModel(VISUALIZATION_DATA.protein_structure, 'pdb');
        }

        const attention = ts.attention_normalized[state.selectedFrameIndex];
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

        proteinViewer.zoomTo();
        proteinViewer.render();
        return;
    }

    // Priority 2: State structure view (average + representatives)
    if (state.selectedState !== null && ts.state_structures) {
        const stateData = ts.state_structures[state.selectedState];
        if (stateData && stateData.average) {
            proteinViewer.removeAllModels();

            // Average structure — color by B-factor (blue→white→red)
            proteinViewer.addModel(stateData.average, 'pdb');
            proteinViewer.setStyle({model: 0}, {
                [representation]: {
                    colorscheme: {prop: 'b', gradient: 'rwb', min: 0, max: 90}
                }
            });

            // Representative structures — low opacity grey
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

// Mouse move handler for embedding viewer
function onEmbeddingMouseMove(event) {
    const rect = embeddingRenderer.domElement.getBoundingClientRect();
    window.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    window.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    window.raycaster.setFromCamera(window.mouse, embeddingCamera);

    // Check active point cloud only
    const activeCloud = pointClouds.find(pc => pc.userData.isActive);
    if (!activeCloud) return;

    const intersects = window.raycaster.intersectObject(activeCloud);

    if (intersects.length > 0) {
        const pointIndex = intersects[0].index;
        state.hoveredFrameIndex = pointIndex;

        const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
        const frameIdx = ts.frame_indices[pointIndex];
        const stateIdx = ts.state_assignments[pointIndex];

        showTooltip(event, `
            Frame: ${frameIdx}<br/>
            State: ${stateIdx}<br/>
            X: ${ts.embeddings[pointIndex][0].toFixed(3)}<br/>
            Y: ${ts.embeddings[pointIndex][1].toFixed(3)}
        `);
    } else {
        state.hoveredFrameIndex = null;
        hideTooltip();
    }
}

// Click handler for embedding viewer
function onEmbeddingClick(event) {
    if (state.hoveredFrameIndex !== null) {
        state.selectedFrameIndex = state.hoveredFrameIndex;
        state.selectedState = null;

        // Update marker
        updateSelectionMarker();

        // Update state chip highlights
        document.querySelectorAll('.state-chip').forEach(c => c.classList.remove('active'));

        // Update protein viewer
        updateProteinViewer();

        console.log(`Selected frame: ${state.selectedFrameIndex}`);
    }
}

// Update selection marker
function updateSelectionMarker() {
    // Remove old marker
    if (selectedMarker) {
        embeddingScene.remove(selectedMarker);
    }

    if (state.selectedFrameIndex === null) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const point = ts.embeddings[state.selectedFrameIndex];

    const bounds = VISUALIZATION_DATA.bounds;
    const scaleX = 10 / (bounds.max_x - bounds.min_x);
    const scaleY = 10 / (bounds.max_y - bounds.min_y);
    const centerX = (bounds.max_x + bounds.min_x) / 2;
    const centerY = (bounds.max_y + bounds.min_y) / 2;

    const x = (point[0] - centerX) * scaleX;
    const y = (point[1] - centerY) * scaleY;
    const z = state.currentTimescaleIndex * VISUALIZATION_DATA.config.embedding.z_spacing;

    const geometry = new THREE.SphereGeometry(0.2, 16, 16);
    const material = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.7
    });
    selectedMarker = new THREE.Mesh(geometry, material);
    selectedMarker.position.set(x, y, z);

    embeddingScene.add(selectedMarker);
}

// Tooltip functions
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

// Initialize state details panel
function initStateDetails() {
    const container = document.getElementById('state-transitions');
    if (!container) return;

    transitionsSvg = d3.select('#state-transitions')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%');
}

// Update state details (selector chips + transitions)
function updateStateDetails(timescale) {
    const selectorContainer = document.getElementById('state-selector');
    if (!selectorContainer) return;

    const stateColors = VISUALIZATION_DATA.config.colors.states;
    selectorContainer.innerHTML = '';

    for (let i = 0; i < timescale.n_states; i++) {
        const chip = document.createElement('span');
        chip.className = 'state-chip';
        if (state.selectedState === i) chip.classList.add('active');
        chip.style.backgroundColor = stateColors[i % stateColors.length];
        chip.textContent = `S${i}`;
        chip.addEventListener('click', () => selectState(i));
        selectorContainer.appendChild(chip);
    }

    // Clamp selectedState if new timescale has fewer states
    if (state.selectedState !== null && state.selectedState >= timescale.n_states) {
        state.selectedState = null;
    }

    updateStateTransitions();
}

// Select/deselect a state
function selectState(stateIndex) {
    if (state.selectedState === stateIndex) {
        // Toggle off
        state.selectedState = null;
    } else {
        state.selectedState = stateIndex;
    }

    // Deselect frame when selecting a state
    state.selectedFrameIndex = null;
    if (selectedMarker) {
        embeddingScene.remove(selectedMarker);
        selectedMarker = null;
    }

    // Update chip highlights
    document.querySelectorAll('.state-chip').forEach((chip, i) => {
        chip.classList.toggle('active', i === state.selectedState);
    });

    updateStateTransitions();
    updateProteinViewer();
}

// Update state transition bar chart
function updateStateTransitions() {
    if (!transitionsSvg) return;
    transitionsSvg.selectAll('*').remove();

    if (state.selectedState === null) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];
    const row = ts.transition_matrix[state.selectedState];
    if (!row) return;

    const stateColors = VISUALIZATION_DATA.config.colors.states;

    // Build data sorted descending by probability
    const data = row.map((prob, i) => ({ state: i, prob: prob }))
        .sort((a, b) => b.prob - a.prob);

    const container = document.getElementById('state-transitions');
    const width = container.clientWidth;
    const height = container.clientHeight;

    transitionsSvg
        .attr('width', width)
        .attr('height', height);

    const margin = { top: 25, right: 50, bottom: 5, left: 35 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const barHeight = Math.min(22, innerH / data.length - 2);

    const g = transitionsSvg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Title
    transitionsSvg.append('text')
        .attr('x', width / 2)
        .attr('y', 16)
        .attr('text-anchor', 'middle')
        .attr('class', 'transition-bar-label')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .text(`Transitions from S${state.selectedState}`);

    const xScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.prob) || 1])
        .range([0, innerW]);

    // Bars
    data.forEach((d, i) => {
        const y = i * (barHeight + 2);

        // Label
        g.append('text')
            .attr('class', 'transition-bar-label')
            .attr('x', -5)
            .attr('y', y + barHeight / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .text(`S${d.state}`);

        // Bar
        g.append('rect')
            .attr('x', 0)
            .attr('y', y)
            .attr('width', xScale(d.prob))
            .attr('height', barHeight)
            .attr('fill', stateColors[d.state % stateColors.length])
            .attr('rx', 3)
            .attr('ry', 3)
            .attr('opacity', 0.85);

        // Value label
        g.append('text')
            .attr('class', 'transition-bar-value')
            .attr('x', xScale(d.prob) + 4)
            .attr('y', y + barHeight / 2)
            .attr('dominant-baseline', 'middle')
            .text(d.prob.toFixed(3));
    });
}

// Window resize handler
function onWindowResize() {
    const embeddingContainer = document.getElementById('embedding-viewer');
    const width = embeddingContainer.clientWidth;
    const height = embeddingContainer.clientHeight;

    embeddingCamera.aspect = width / height;
    embeddingCamera.updateProjectionMatrix();
    embeddingRenderer.setSize(width, height);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    // Update controls
    if (embeddingControls) {
        embeddingControls.update();
    }

    // Render
    if (embeddingRenderer && embeddingScene && embeddingCamera) {
        embeddingRenderer.render(embeddingScene, embeddingCamera);
    }
}

// Initialize event listeners
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
        });
    }

    // Camera rotation toggle
    const rotationToggle = document.getElementById('camera-rotation');
    if (rotationToggle) {
        rotationToggle.checked = state.camera.rotation;
        rotationToggle.addEventListener('change', (e) => {
            state.camera.rotation = e.target.checked;
        });
    }
}

// =============================================
// Diagnostics panel
// =============================================

function initDiagnosticsPanel() {
    // Check if any timescale has diagnostic data
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

    // Update sidebar badge
    const badge = document.getElementById('diag-recommendation-badge');
    if (badge) {
        const rec = report.recommendation || 'keep';
        badge.textContent = rec;
        badge.className = 'diag-badge ' + rec;
    }

    // Quick info in sidebar
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
            lines.push(`Merge groups: ${groups}`);
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
    const merge = diag.report ? diag.report.merge : null;
    const plots = diag.plots || {};

    // Update title
    const title = document.getElementById('diagnostics-title');
    if (title) title.textContent = `State Diagnostics — Lagtime ${ts.lagtime}`;

    // Recommendation banner
    const banner = document.getElementById('diagnostics-banner');
    if (banner) {
        const rec = report.recommendation || 'keep';
        banner.className = 'diagnostics-banner ' + rec;
        let bannerText = '';
        if (rec === 'keep') {
            bannerText = `All ${report.original_n_states} states are well-separated. No merging needed.`;
        } else if (rec === 'merge') {
            bannerText = `Recommend merging: ${report.original_n_states} → ${report.effective_n_states} states.`;
            if (merge) {
                bannerText += ` VAMP-2 drop: ${(merge.vamp2_drop * 100).toFixed(1)}%`;
                bannerText += merge.validation_passed ? ' (passed)' : ' (FAILED)';
            }
        } else if (rec === 'retrain') {
            bannerText = `Large reduction detected (${report.original_n_states} → ${report.effective_n_states}). Retraining recommended.`;
        }
        banner.textContent = bannerText;
    }

    // Set plot images and show/hide tabs based on availability
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

    // Populate report details table in summary tab
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
            html += `<tr><td>Merge groups</td><td>${groups}</td></tr>`;
        }
        if (merge) {
            html += `<tr><td>VAMP-2 original</td><td>${merge.vamp2_original ? merge.vamp2_original.toFixed(4) : '—'}</td></tr>`;
            html += `<tr><td>VAMP-2 merged</td><td>${merge.vamp2_merged ? merge.vamp2_merged.toFixed(4) : '—'}</td></tr>`;
            html += `<tr><td>VAMP-2 drop</td><td>${merge.vamp2_drop ? (merge.vamp2_drop * 100).toFixed(1) + '%' : '—'}</td></tr>`;
            html += `<tr><td>Validation</td><td>${merge.validation_passed ? 'PASSED' : 'FAILED'}</td></tr>`;
            if (merge.state_mapping) {
                const mapping = Object.entries(merge.state_mapping)
                    .map(([k, v]) => `S${k} ← {${v.join(',')}}`)
                    .join(', ');
                html += `<tr><td>State mapping</td><td>${mapping}</td></tr>`;
            }
        }
        html += '</table>';
        details.innerHTML = html;
    }

    // Activate first available tab
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

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
