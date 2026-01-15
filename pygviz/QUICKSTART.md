# Quick Start Guide

This guide will get you up and running with MD Visualizer in 5 minutes.

## Installation

### Step 1: Install Dependencies

```bash
pip install -r md_visualizer/requirements.txt
```

Minimum requirements:
- `numpy >= 1.20.0`
- `jinja2 >= 3.0.0`

Optional (for trajectory handling):
- `mdtraj >= 1.9.0`

### Step 2: Verify Installation

Test with mock data:

```bash
cd md_visualizer/examples
python generate_mock_data.py
```

This will create `output/mock_visualization.html`.

Open the HTML file in your browser to see the interactive visualization!

## Basic Usage

### 1. Import the Package

```python
from md_visualizer import MDTrajectoryVisualizer
import numpy as np
```

### 2. Create Visualizer

```python
viz = MDTrajectoryVisualizer()
```

### 3. Add Your Data

```python
# Your MD analysis results
embeddings = np.load("embeddings.npy")        # Shape: (n_frames, 2)
states = np.load("states.npy")                # Shape: (n_frames,)
trans_matrix = np.load("transition.npy")      # Shape: (n_states, n_states)
attention = np.load("attention.npy")          # Shape: (n_frames, n_residues)
frame_indices = np.arange(len(embeddings))

# Add timescale
viz.add_timescale(
    lagtime=10,
    embeddings=embeddings,
    frame_indices=frame_indices,
    state_assignments=states,
    transition_matrix=trans_matrix,
    attention_values=attention
)
```

### 4. Add Protein Structure

```python
# Option 1: From PDB file
viz.set_protein_structure(pdb_path="protein.pdb")

# Option 2: From PDB ID (downloads from RCSB)
viz.set_protein_structure(pdb_path="1UBQ")

# Option 3: From trajectory
viz.set_protein_structure(
    trajectory_path="trajectory.xtc",
    topology_path="topology.pdb",
    frame_index=0
)
```

### 5. Generate Visualization

```python
# Save to file
viz.generate("my_analysis.html", title="My MD Analysis")

# Or launch directly in browser
viz.show()
```

## Complete Example

```python
from md_visualizer import MDTrajectoryVisualizer
import numpy as np

# Initialize
viz = MDTrajectoryVisualizer()

# Add data for multiple timescales
for lagtime in [1, 5, 10, 50]:
    embeddings = load_embeddings(lagtime)      # Your loading function
    states = load_states(lagtime)
    trans_matrix = load_transition(lagtime)
    attention = load_attention(lagtime)

    viz.add_timescale(
        lagtime=lagtime,
        embeddings=embeddings,
        frame_indices=np.arange(len(embeddings)),
        state_assignments=states,
        transition_matrix=trans_matrix,
        attention_values=attention
    )

# Add structure
viz.set_protein_structure(pdb_path="protein.pdb")

# Generate
viz.generate("output.html")
```

## What You'll See

The visualization has four main panels:

1. **Left Sidebar**: Controls for selecting timescales and adjusting view settings
2. **Center**: 3D visualization of embeddings (one layer per timescale)
3. **Right**: Protein structure with attention-based coloring
4. **Bottom**: Transition matrix heatmap

### Interactive Features

- **Click** points to select frames
- **Drag** to rotate views
- **Scroll** to zoom
- **Hover** for details
- **Switch** between timescales
- **Toggle** attention coloring

## Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- See [example_usage.py](md_visualizer/examples/example_usage.py) for more examples
- Customize with configuration options (theme, colors, sizes)
- Export frames and transition graphs

## Troubleshooting

**Can't find module?**
```bash
# Make sure you're in the right directory
cd PyGVAMP-visualize
python -c "import md_visualizer; print('Success!')"
```

**Visualization doesn't load?**
- Check browser console (F12) for errors
- Ensure internet connection (for CDN libraries)
- Try Chrome or Firefox

**Need help?**
- Check the [README.md](README.md) for detailed documentation
- Look at the examples in `md_visualizer/examples/`

## Integration with PyGVAMP

If you're using PyGVAMP, the typical workflow is:

1. Run PyGVAMP analysis to get embeddings, attention, and states
2. Use this visualizer to create interactive 3D visualization
3. Explore your results in the browser

See [example_usage.py](md_visualizer/examples/example_usage.py) for integration examples.

Happy visualizing! 🎨
