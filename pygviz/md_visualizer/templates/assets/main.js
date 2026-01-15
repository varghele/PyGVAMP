// MD Trajectory Visualization - Main JavaScript
// This file is rendered as a Jinja2 template

// Global state
const state = {
    currentTimescaleIndex: 0,
    selectedFrameIndex: null,
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

    // Reset selection
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

            // Add text if cell is large enough
            if (cellSize > 30 && value > 0.01) {
                g.append('text')
                    .attr('class', 'matrix-value')
                    .attr('x', j * cellSize + cellSize / 2)
                    .attr('y', i * cellSize + cellSize / 2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .text(value.toFixed(2));
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
    if (!VISUALIZATION_DATA.protein_structure || !proteinViewer) return;

    const ts = VISUALIZATION_DATA.timescales[state.currentTimescaleIndex];

    if (state.selectedFrameIndex !== null && state.showAttention) {
        // Get attention values for selected frame
        const attention = ts.attention_normalized[state.selectedFrameIndex];

        // Color by attention
        const colorScale = d3.scaleLinear()
            .domain([0, 1])
            .range([
                VISUALIZATION_DATA.config.colors.attention.low,
                VISUALIZATION_DATA.config.colors.attention.high
            ]);

        proteinViewer.setStyle({}, {});

        // Color each residue by attention
        attention.forEach((value, residueIndex) => {
            const color = colorScale(value);
            const representation = VISUALIZATION_DATA.config.protein.representation;

            proteinViewer.setStyle(
                { resi: residueIndex + 1 },
                { [representation]: { color: color } }
            );
        });
    } else {
        // Default coloring
        const representation = VISUALIZATION_DATA.config.protein.representation;
        proteinViewer.setStyle({}, { [representation]: { color: 'spectrum' } });
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

        // Update marker
        updateSelectionMarker();

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

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
