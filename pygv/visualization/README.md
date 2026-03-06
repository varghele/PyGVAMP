# MD Visualizer

Interactive 3D visualization toolkit for molecular dynamics trajectory analysis, designed for integration with PyGVAMP and other MD analysis tools.

## Features

- **3D Embedding Visualization**: Interactive Three.js-based visualization of 2D embeddings in 3D space, with multiple timescales displayed as layers
- **Protein Structure Viewer**: 3Dmol.js-powered protein visualization with attention-based coloring
- **Transition Matrices**: D3.js heatmaps showing state transition probabilities
- **Multi-timescale Analysis**: Support for analyzing dynamics at different lagtimes
- **Interactive**: Click points to select frames, hover for details, rotate and zoom views
- **Standalone HTML**: Generate self-contained HTML files that work in any modern browser
- **Customizable**: Extensive configuration options for colors, styles, and visualization parameters

## Installation

### Basic Installation

```bash
# Clone or download the repository
cd PyGVAMP-visualize

# Install dependencies
pip install -r md_visualizer/requirements.txt
pip install numpy jinja2
```

### Dependencies

Core requirements:
- `numpy >= 1.20.0`
- `jinja2 >= 3.0.0`
- `mdtraj >= 1.9.0` (for trajectory handling)

Optional:
- `networkx >= 2.6.0` (for graph export features)
- `MDAnalysis >= 2.0.0` (alternative to MDTraj)

JavaScript libraries (loaded via CDN):
- Three.js (3D graphics)
- 3Dmol.js (protein visualization)
- D3.js (data visualization)

## Quick Start

### 1. Test with Mock Data

Generate a visualization with synthetic data to test the installation:

```bash
cd md_visualizer/examples
python generate_mock_data.py
```

This creates `output/mock_visualization.html`. Open it in a browser to see the interactive visualization.

To launch directly in browser:
```bash
python generate_mock_data.py --show
```

### 2. Basic Usage with Your Data

```python
from md_visualizer import MDTrajectoryVisualizer
import numpy as np

# Initialize visualizer
viz = MDTrajectoryVisualizer()

# Add data for one timescale
viz.add_timescale(
    lagtime=10,
    embeddings=embeddings_2d,      # numpy array (n_frames, 2)
    frame_indices=frame_indices,    # numpy array (n_frames,)
    state_assignments=states,       # numpy array (n_frames,)
    transition_matrix=trans_matrix, # numpy array (n_states, n_states)
    attention_values=attention      # numpy array (n_frames, n_residues)
)

# Add protein structure
viz.set_protein_structure(pdb_path="protein.pdb")

# Generate visualization
viz.generate("my_analysis.html", title="MD Trajectory Analysis")

# Or launch directly in browser
viz.show()
```

## API Reference

### MDTrajectoryVisualizer

Main class for creating visualizations.

#### `__init__(config=None)`

Initialize the visualizer with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary

**Configuration options:**
```python
config = {
    'theme': 'dark',  # 'dark' or 'light'
    'colors': {
        'states': ['#FF6B6B', '#4ECDC4', '#45B7D1', ...],  # State colors
        'attention': {
            'low': '#0000FF',   # Low attention color
            'high': '#FF0000'   # High attention color
        }
    },
    'embedding': {
        'point_size': 0.1,      # Size of points in 3D view
        'z_spacing': 5.0        # Spacing between timescale layers
    },
    'protein': {
        'representation': 'cartoon',  # 'cartoon', 'surface', 'stick', etc.
        'color_scheme': 'attention'   # 'attention' or 'structure'
    }
}
```

#### `add_timescale(...)`

Add data for one timescale/lagtime.

**Parameters:**
- `lagtime` (int): Lagtime value
- `embeddings` (np.ndarray): Shape (n_frames, 2) - 2D embeddings
- `frame_indices` (np.ndarray): Shape (n_frames,) - Frame indices
- `state_assignments` (np.ndarray): Shape (n_frames,) - State labels
- `transition_matrix` (np.ndarray): Shape (n_states, n_states) - Transition probabilities
- `attention_values` (np.ndarray): Shape (n_frames, n_residues) - Attention weights
- `metadata` (dict, optional): Additional metadata

**Raises:**
- `ValueError`: If data validation fails

#### `set_protein_structure(...)`

Set protein structure for visualization.

**Parameters (choose one):**
- `pdb_path` (str): Path to PDB file or 4-letter PDB ID (e.g., '1UBQ')
- `pdb_string` (str): PDB content as string
- `trajectory_path` + `topology_path` (str): Extract from trajectory
  - `frame_index` (int): Frame to extract (default: 0)

**Example:**
```python
# From file
viz.set_protein_structure(pdb_path="protein.pdb")

# From RCSB database
viz.set_protein_structure(pdb_path="1UBQ")

# From trajectory
viz.set_protein_structure(
    trajectory_path="traj.xtc",
    topology_path="topology.pdb",
    frame_index=0
)
```

#### `generate(output_path, title="MD Visualization", standalone=True)`

Generate HTML file.

**Parameters:**
- `output_path` (str): Where to save HTML file
- `title` (str): Page title
- `standalone` (bool): Include all assets (currently always uses CDN)

#### `show(port=8000, auto_open=True)`

Launch local server and open in browser.

**Parameters:**
- `port` (int): Server port
- `auto_open` (bool): Open browser automatically

#### `to_json()`

Export all data as JSON string.

**Returns:** JSON string with all visualization data

#### `export_selected_frames(frame_indices, output_dir, trajectory_path, topology_path)`

Export selected frames as PDB files.

**Parameters:**
- `frame_indices` (np.ndarray): Frames to export
- `output_dir` (str): Output directory
- `trajectory_path` (str): Path to trajectory
- `topology_path` (str): Path to topology

#### `export_transition_graph(lagtime, output_path, format='graphml')`

Export transition matrix as graph file.

**Parameters:**
- `lagtime` (int): Which lagtime to export
- `output_path` (str): Output file path
- `format` (str): 'graphml', 'gml', or 'dot'

**Requires:** `networkx`

#### `get_summary()`

Get summary of loaded data.

**Returns:** Formatted summary string

## Data Format Requirements

### Embeddings
- **Shape:** `(n_frames, 2)`
- **Type:** `numpy.ndarray` of float
- **Description:** 2D projection of molecular dynamics data
- **Constraints:** No NaN or Inf values

### Frame Indices
- **Shape:** `(n_frames,)`
- **Type:** `numpy.ndarray` of int
- **Description:** Original frame numbers from trajectory
- **Constraints:** Non-negative integers

### State Assignments
- **Shape:** `(n_frames,)`
- **Type:** `numpy.ndarray` of int
- **Description:** Discrete state labels for each frame
- **Constraints:** Non-negative integers, max value < n_states

### Transition Matrix
- **Shape:** `(n_states, n_states)`
- **Type:** `numpy.ndarray` of float
- **Description:** Transition probabilities between states
- **Constraints:**
  - Values in [0, 1]
  - Each row should sum to 1

### Attention Values
- **Shape:** `(n_frames, n_residues)`
- **Type:** `numpy.ndarray` of float
- **Description:** Attention weights for each residue in each frame
- **Constraints:** No NaN or Inf values
- **Note:** Will be automatically normalized to [0, 1]

## Usage Examples

### Multi-timescale Analysis

```python
viz = MDTrajectoryVisualizer()

for lagtime in [1, 5, 10, 50, 100]:
    # Load data for this lagtime
    embeddings = load_embeddings(lagtime)
    states = load_states(lagtime)
    trans_matrix = load_transition_matrix(lagtime)
    attention = load_attention(lagtime)
    frame_indices = load_frame_indices(lagtime)

    viz.add_timescale(
        lagtime=lagtime,
        embeddings=embeddings,
        frame_indices=frame_indices,
        state_assignments=states,
        transition_matrix=trans_matrix,
        attention_values=attention
    )

viz.set_protein_structure(pdb_path="protein.pdb")
viz.generate("multi_timescale.html")
```

### Custom Configuration

```python
config = {
    'theme': 'light',
    'colors': {
        'states': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
        'attention': {'low': '#ffffff', 'high': '#000000'}
    },
    'embedding': {
        'point_size': 0.2,
        'z_spacing': 10.0
    }
}

viz = MDTrajectoryVisualizer(config=config)
# ... add data ...
viz.generate("custom_style.html")
```

### Integration with PyGVAMP

```python
from md_visualizer import MDTrajectoryVisualizer
# from pygvamp import GraphVAMP  # Your PyGVAMP import

# Train model (example)
# model = GraphVAMP(n_components=2, lagtime=10)
# model.fit(graphs)

# Get results
# embeddings = model.transform(graphs)
# attention = model.get_attention_weights()
# states = model.cluster(embeddings, n_clusters=5)
# trans_matrix = model.estimate_transition_matrix(states)

# Create visualization
viz = MDTrajectoryVisualizer()
viz.add_timescale(
    lagtime=10,
    embeddings=embeddings,
    frame_indices=np.arange(len(embeddings)),
    state_assignments=states,
    transition_matrix=trans_matrix,
    attention_values=attention
)

viz.set_protein_structure(
    trajectory_path="trajectory.xtc",
    topology_path="topology.pdb"
)

viz.generate("pygvamp_analysis.html")
```

## Interactive Features

Once you open the HTML visualization in a browser:

### Embedding Viewer (Left Panel)
- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan view
- **Scroll**: Zoom in/out
- **Click Point**: Select frame (updates protein viewer)
- **Hover Point**: Show frame details

### Controls (Sidebar)
- **Timescale Buttons**: Switch between different lagtimes
- **Attention Toggle**: Show/hide attention coloring on protein
- **Protein Style**: Change representation (cartoon, surface, stick, etc.)
- **State Legend**: See frame counts per state

### Protein Viewer (Right Panel)
- **Drag**: Rotate protein
- **Scroll**: Zoom
- **Colors**: Blue (low attention) → Red (high attention)

### Transition Matrix (Bottom)
- **Hover Cells**: See transition probabilities
- **Color Intensity**: Darker = higher probability

## Project Structure

```
md_visualizer/
├── __init__.py              # Package initialization
├── visualizer.py            # Main MDTrajectoryVisualizer class
├── data_handler.py          # Data validation and processing
├── templates/
│   ├── index.html           # Jinja2 HTML template
│   └── assets/
│       ├── style.css        # Styling
│       └── main.js          # JavaScript visualization logic
├── examples/
│   ├── generate_mock_data.py   # Mock data generator
│   └── example_usage.py        # Usage examples
└── requirements.txt         # Python dependencies
```

## Troubleshooting

### Common Issues

**"Module 'mdtraj' not found"**
```bash
pip install mdtraj
```

**Protein structure doesn't match attention dimensions**
- Ensure PDB has same number of residues as attention array second dimension
- Check with `DataProcessor.get_residue_count_from_pdb(pdb_string)`

**Visualization doesn't load in browser**
- Check browser console for JavaScript errors
- Ensure you have internet connection (for CDN libraries)
- Try a different modern browser (Chrome, Firefox, Edge)

**Points not visible in 3D view**
- Adjust `point_size` in config
- Check that embeddings are in reasonable range
- Try zooming out in the viewer

## Advanced Features

### Export Frame Structures

```python
# Export specific frames as PDB files
selected_frames = np.array([0, 10, 20, 50])
viz.export_selected_frames(
    frame_indices=selected_frames,
    output_dir="frames/",
    trajectory_path="traj.xtc",
    topology_path="topology.pdb"
)
```

### Export Transition Network

```python
# Export as GraphML for network analysis
viz.export_transition_graph(
    lagtime=10,
    output_path="network.graphml",
    format='graphml'
)

# Load in NetworkX
import networkx as nx
G = nx.read_graphml("network.graphml")
```

### Custom Data Processing

```python
from md_visualizer import DataProcessor

# Normalize attention values
attention_norm = DataProcessor.normalize_attention(
    attention_values,
    method='minmax'  # or 'zscore'
)

# Validate data
embeddings = DataProcessor.validate_embeddings(embeddings_2d)
trans_matrix = DataProcessor.validate_transition_matrix(trans_matrix)
```

## Contributing

Contributions are welcome! Areas for enhancement:

- Additional protein representations
- Animation of trajectories
- More export formats
- Jupyter notebook integration
- Additional clustering visualizations

## License

See LICENSE file for details.

## Citation

If you use this visualization toolkit in your research, please cite:

```
[Citation information to be added]
```

## Acknowledgments

Built with:
- [Three.js](https://threejs.org/) - 3D graphics
- [3Dmol.js](https://3dmol.csb.pitt.edu/) - Molecular visualization
- [D3.js](https://d3js.org/) - Data visualization
- [MDTraj](https://mdtraj.org/) - Trajectory analysis
- [Jinja2](https://jinja.palletsprojects.com/) - Templating

Designed for integration with PyGVAMP molecular dynamics analysis.
