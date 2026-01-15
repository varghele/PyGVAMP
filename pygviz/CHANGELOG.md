# Changelog

All notable changes to MD Visualizer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-12

### Added
- Initial release of MD Visualizer
- Core `MDTrajectoryVisualizer` class for creating interactive visualizations
- `DataProcessor` class for data validation and processing
- Support for multi-timescale trajectory analysis
- Three.js-based 3D embedding visualization
- 3Dmol.js protein structure viewer with attention mapping
- D3.js transition matrix heatmaps
- Interactive features:
  - Click to select frames
  - Hover for tooltips
  - Rotate and zoom views
  - Switch between timescales
  - Toggle attention coloring
- Configuration system for themes, colors, and visualization parameters
- Data export features:
  - Export selected frames as PDB files
  - Export transition graphs (GraphML, GML, DOT)
  - Export data as JSON
- Comprehensive documentation:
  - README with full API reference
  - Quick start guide
  - Usage examples
  - Integration examples with PyGVAMP
- Mock data generator for testing
- Template examples for real data integration
- Jinja2-based HTML templating system
- Standalone HTML generation (all JavaScript via CDN)
- Dark and light theme support
- Type hints throughout codebase
- Detailed docstrings for all public methods

### Features by Component

#### Visualizer (`visualizer.py`)
- `add_timescale()` - Add data for multiple lagtimes
- `set_protein_structure()` - Load PDB from file, ID, or trajectory
- `generate()` - Create HTML visualization file
- `show()` - Launch local server and open in browser
- `to_json()` - Export all data as JSON
- `export_selected_frames()` - Save frames as PDB files
- `export_transition_graph()` - Export as network graph
- `get_summary()` - Print data summary

#### Data Handler (`data_handler.py`)
- Validation functions for all data types
- Attention value normalization (minmax, zscore)
- Trajectory frame extraction (MDTraj)
- PDB loading from file or RCSB
- JSON serialization for visualization
- Embedding bounds computation
- Residue count extraction

#### Visualization Components
- 3D point clouds with state-based coloring
- Multiple timescale layers with z-spacing
- Interactive protein viewer with attention coloring
- Transition matrix with probability heatmap
- State legend with frame counts
- Attention color scale
- Tooltips and info displays

### Dependencies
- numpy >= 1.20.0
- jinja2 >= 3.0.0
- mdtraj >= 1.9.0 (optional)
- networkx >= 2.6.0 (optional)

### Browser Compatibility
- Tested on modern browsers (Chrome, Firefox, Edge, Safari)
- Requires JavaScript enabled
- Requires internet connection for CDN libraries

### Known Limitations
- Large datasets (>10,000 frames) may be slow to render
- JavaScript libraries loaded from CDN (no offline mode yet)
- Limited to 2D embeddings (will add 3D support in future)
- Protein coloring requires matching residue count with attention dimensions

### Future Enhancements
- Trajectory animation playback
- Jupyter notebook integration
- Additional protein representations
- Offline mode with bundled JavaScript
- 3D embedding support
- State transition pathway visualization
- Frame-to-frame difference highlighting
- Export as static images/videos
- Custom colormaps
- Clustering visualization overlays

## [Unreleased]

### Planned
- Jupyter notebook widget support
- Animation controls for trajectory playback
- Additional export formats
- Performance optimizations for large datasets
- Bundled JavaScript option for offline use
- Extended documentation with video tutorials
- Unit tests and CI/CD integration
