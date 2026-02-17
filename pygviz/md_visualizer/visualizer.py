"""Main visualization class for MD trajectory analysis."""

import os
import json
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
from typing import Optional, Dict, List, Union
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .data_handler import DataProcessor


class MDTrajectoryVisualizer:
    """
    Interactive 3D visualization toolkit for molecular dynamics trajectory analysis.

    This class enables creation of interactive web-based visualizations combining:
    - 2D embeddings in 3D space (one layer per timescale)
    - Protein structure visualization with attention mapping
    - Transition matrices and state information
    - Multi-timescale analysis

    Examples
    --------
    >>> pygviz = MDTrajectoryVisualizer()
    >>> pygviz.add_timescale(
    ...     lagtime=10,
    ...     embeddings=embeddings_2d,
    ...     frame_indices=frame_indices,
    ...     state_assignments=states,
    ...     transition_matrix=trans_matrix,
    ...     attention_values=attention
    ... )
    >>> pygviz.set_protein_structure(pdb_path="protein.pdb")
    >>> pygviz.generate("output.html")
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualizer with optional configuration.

        Parameters
        ----------
        config : Dict, optional
            Configuration dictionary with visualization settings.
            Default settings will be used if not provided.

            Supported configuration keys:
            - 'theme': 'dark' or 'light' (default: 'dark')
            - 'colors': Dictionary of color schemes
                - 'states': List of colors for states
                - 'attention': Dict with 'low' and 'high' colors
            - 'embedding': Dictionary of embedding settings
                - 'point_size': Size of points (default: 0.1)
                - 'z_spacing': Spacing between timescale layers (default: 5.0)
            - 'protein': Dictionary of protein visualization settings
                - 'representation': 'cartoon', 'surface', or 'stick' (default: 'cartoon')
                - 'color_scheme': 'attention' or 'structure' (default: 'attention')
        """
        self.timescales_data = []
        self.protein_structure = None
        self.protein_source = None

        # Default configuration
        self.config = {
            'theme': 'dark',
            'colors': {
                'states': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                          '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AAB7B8'],
                'attention': {'low': '#0000FF', 'high': '#FF0000'}
            },
            'embedding': {
                'point_size': 0.3,  # Increased for easier clicking
                'z_spacing': 5.0
            },
            'protein': {
                'representation': 'cartoon',
                'color_scheme': 'attention'
            }
        }

        # Update with user config
        if config:
            self._update_config(config)

        # Get template directory
        self.template_dir = Path(__file__).parent / 'templates'

    def _update_config(self, config: Dict):
        """Recursively update configuration."""
        def update_recursive(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    update_recursive(base[key], value)
                else:
                    base[key] = value

        update_recursive(self.config, config)

    def add_timescale(
        self,
        lagtime: int,
        embeddings: np.ndarray,
        frame_indices: np.ndarray,
        state_assignments: np.ndarray,
        transition_matrix: np.ndarray,
        attention_values: np.ndarray,
        metadata: Optional[Dict] = None,
        state_structures: Optional[Dict] = None
    ):
        """
        Add data for one timescale/lagtime.

        Parameters
        ----------
        lagtime : int
            Lagtime value for this timescale
        embeddings : np.ndarray
            2D embeddings array of shape (n_frames, 2)
        frame_indices : np.ndarray
            Frame indices array of shape (n_frames,)
        state_assignments : np.ndarray
            State assignments array of shape (n_frames,)
        transition_matrix : np.ndarray
            Transition matrix of shape (n_states, n_states)
        attention_values : np.ndarray
            Attention values array of shape (n_frames, n_residues)
        metadata : Dict, optional
            Additional metadata for this timescale

        Raises
        ------
        ValueError
            If any input data fails validation
        """
        # Validate all inputs
        embeddings = DataProcessor.validate_embeddings(embeddings, expected_dims=2)
        n_frames = len(embeddings)

        frame_indices = DataProcessor.validate_frame_indices(frame_indices, n_frames)
        state_assignments = DataProcessor.validate_state_assignments(state_assignments, n_frames)
        transition_matrix = DataProcessor.validate_transition_matrix(transition_matrix)
        attention_values = DataProcessor.validate_attention_values(attention_values, n_frames)

        # Normalize attention values
        attention_normalized = DataProcessor.normalize_attention(attention_values)

        # Check consistency
        n_states = transition_matrix.shape[0]
        max_state = np.max(state_assignments)
        if max_state >= n_states:
            raise ValueError(f"State assignments contain state {max_state} but transition matrix "
                           f"only has {n_states} states")

        # Store data
        timescale_data = {
            'lagtime': int(lagtime),
            'embeddings': embeddings,
            'frame_indices': frame_indices,
            'state_assignments': state_assignments,
            'transition_matrix': transition_matrix,
            'attention_values': attention_values,
            'attention_normalized': attention_normalized,
            'n_states': n_states,
            'n_frames': n_frames,
            'n_residues': attention_values.shape[1],
            'metadata': metadata or {},
            'state_structures': state_structures or {}
        }

        self.timescales_data.append(timescale_data)

        # Sort by lagtime
        self.timescales_data.sort(key=lambda x: x['lagtime'])

    def set_protein_structure(
        self,
        pdb_path: Optional[str] = None,
        pdb_string: Optional[str] = None,
        trajectory_path: Optional[str] = None,
        topology_path: Optional[str] = None,
        frame_index: int = 0
    ):
        """
        Set protein structure data.

        Parameters
        ----------
        pdb_path : str, optional
            Path to PDB file or 4-letter PDB ID (e.g., '1UBQ')
        pdb_string : str, optional
            PDB file contents as string
        trajectory_path : str, optional
            Path to trajectory file (requires topology_path)
        topology_path : str, optional
            Path to topology file (used with trajectory_path)
        frame_index : int, default=0
            Frame index to extract from trajectory

        Raises
        ------
        ValueError
            If invalid combination of parameters is provided
        """
        if pdb_string:
            self.protein_structure = pdb_string
            self.protein_source = 'string'

        elif pdb_path:
            self.protein_structure = DataProcessor.load_pdb_file(pdb_path)
            self.protein_source = pdb_path

        elif trajectory_path and topology_path:
            pdb_strings = DataProcessor.extract_frames_from_trajectory(
                trajectory_path, topology_path, np.array([frame_index])
            )
            self.protein_structure = pdb_strings[0]
            self.protein_source = f"{trajectory_path} (frame {frame_index})"

        else:
            raise ValueError("Must provide either pdb_path, pdb_string, or "
                           "both trajectory_path and topology_path")

        # Validate residue count matches attention values
        if self.timescales_data:
            n_residues_pdb = DataProcessor.get_residue_count_from_pdb(self.protein_structure)
            expected_residues = self.timescales_data[0]['n_residues']

            if n_residues_pdb != expected_residues:
                import warnings
                warnings.warn(
                    f"PDB has {n_residues_pdb} residues but attention values have "
                    f"{expected_residues} dimensions. Visualization may be incorrect."
                )

    def to_json(self) -> str:
        """
        Export all data as JSON for the JavaScript frontend.

        Returns
        -------
        str
            JSON string containing all visualization data
        """
        if not self.timescales_data:
            raise ValueError("No timescale data added. Use add_timescale() first.")

        # Compute global bounds for consistent scaling
        all_embeddings = [ts['embeddings'] for ts in self.timescales_data]
        bounds = DataProcessor.compute_embedding_bounds(all_embeddings)

        # Prepare data structure
        export_data = {
            'timescales': [],
            'bounds': bounds,
            'config': self.config,
            'protein_structure': self.protein_structure,
            'protein_source': self.protein_source
        }

        # Add each timescale
        for ts_data in self.timescales_data:
            export_data['timescales'].append({
                'lagtime': ts_data['lagtime'],
                'embeddings': ts_data['embeddings'],
                'frame_indices': ts_data['frame_indices'],
                'state_assignments': ts_data['state_assignments'],
                'transition_matrix': ts_data['transition_matrix'],
                'attention_values': ts_data['attention_values'],
                'attention_normalized': ts_data['attention_normalized'],
                'n_states': ts_data['n_states'],
                'n_frames': ts_data['n_frames'],
                'n_residues': ts_data['n_residues'],
                'metadata': ts_data['metadata'],
                'state_structures': ts_data['state_structures']
            })

        return DataProcessor.prepare_json_data(export_data)

    def generate(
        self,
        output_path: str,
        title: str = "MD Trajectory Visualization",
        standalone: bool = True
    ):
        """
        Generate HTML file with all visualizations.

        Parameters
        ----------
        output_path : str
            Path where HTML file will be saved
        title : str, default="MD Trajectory Visualization"
            Title for the visualization page
        standalone : bool, default=True
            If True, include all assets inline (currently always uses CDN)

        Raises
        ------
        ValueError
            If no data has been added to the visualizer
        """
        if not self.timescales_data:
            raise ValueError("No timescale data added. Use add_timescale() first.")

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2 environment
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Load template
        template = env.get_template('index.html')

        # Get JSON data
        data_json = self.to_json()

        # Render template
        html_content = template.render(
            title=title,
            data_json=data_json,
            standalone=standalone
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Visualization saved to: {output_path.absolute()}")

    def show(self, port: int = 8000, auto_open: bool = True):
        """
        Launch local server and open visualization in browser.

        Parameters
        ----------
        port : int, default=8000
            Port number for local server
        auto_open : bool, default=True
            If True, automatically open browser

        Raises
        ------
        ValueError
            If no data has been added to the visualizer
        """
        import tempfile

        # Generate temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            temp_path = f.name

        self.generate(temp_path, standalone=True)

        # Start simple HTTP server
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(Path(temp_path).parent), **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress server logs

        def start_server():
            with socketserver.TCPServer(("", port), Handler) as httpd:
                httpd.serve_forever()

        # Start server in background thread
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()

        # Open browser
        url = f"http://localhost:{port}/{Path(temp_path).name}"
        if auto_open:
            print(f"Opening visualization at {url}")
            print("Press Ctrl+C to stop the server")
            webbrowser.open(url)
        else:
            print(f"Server running at {url}")
            print("Press Ctrl+C to stop the server")

        # Keep main thread alive
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\nServer stopped")

    def export_selected_frames(
        self,
        frame_indices: np.ndarray,
        output_dir: str,
        trajectory_path: str,
        topology_path: str
    ):
        """
        Export selected frames as individual PDB files.

        Parameters
        ----------
        frame_indices : np.ndarray
            Indices of frames to export
        output_dir : str
            Directory where PDB files will be saved
        trajectory_path : str
            Path to trajectory file
        topology_path : str
            Path to topology file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdb_strings = DataProcessor.extract_frames_from_trajectory(
            trajectory_path, topology_path, frame_indices
        )

        for i, (frame_idx, pdb_string) in enumerate(zip(frame_indices, pdb_strings)):
            output_path = output_dir / f"frame_{frame_idx}.pdb"
            with open(output_path, 'w') as f:
                f.write(pdb_string)

        print(f"Exported {len(frame_indices)} frames to {output_dir.absolute()}")

    def export_transition_graph(
        self,
        lagtime: int,
        output_path: str,
        format: str = 'graphml'
    ):
        """
        Export transition matrix as graph file.

        Parameters
        ----------
        lagtime : int
            Lagtime of the transition matrix to export
        output_path : str
            Path for output file
        format : str, default='graphml'
            Output format: 'graphml', 'gml', or 'dot'

        Raises
        ------
        ValueError
            If lagtime not found or format not supported
        ImportError
            If networkx is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for graph export. "
                            "Install with: pip install networkx")

        # Find timescale data
        ts_data = None
        for ts in self.timescales_data:
            if ts['lagtime'] == lagtime:
                ts_data = ts
                break

        if ts_data is None:
            raise ValueError(f"No data found for lagtime {lagtime}")

        # Create directed graph
        trans_matrix = ts_data['transition_matrix']
        n_states = ts_data['n_states']

        G = nx.DiGraph()

        # Add nodes
        for i in range(n_states):
            G.add_node(i, state=i)

        # Add edges (only for significant transitions)
        threshold = 0.01  # Minimum transition probability
        for i in range(n_states):
            for j in range(n_states):
                if trans_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=float(trans_matrix[i, j]))

        # Export
        output_path = Path(output_path)
        if format == 'graphml':
            nx.write_graphml(G, output_path)
        elif format == 'gml':
            nx.write_gml(G, output_path)
        elif format == 'dot':
            nx.drawing.nx_pydot.write_dot(G, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Transition graph saved to: {output_path.absolute()}")

    def get_summary(self) -> str:
        """
        Get summary of current visualization data.

        Returns
        -------
        str
            Formatted summary string
        """
        if not self.timescales_data:
            return "No data loaded"

        summary = []
        summary.append("=" * 60)
        summary.append("MD Trajectory Visualization Summary")
        summary.append("=" * 60)
        summary.append(f"Number of timescales: {len(self.timescales_data)}")
        summary.append(f"Protein structure: {'Loaded' if self.protein_structure else 'Not loaded'}")
        if self.protein_source:
            summary.append(f"  Source: {self.protein_source}")
        summary.append("")
        summary.append("Timescales:")

        for ts in self.timescales_data:
            summary.append(f"  Lagtime {ts['lagtime']:3d}: "
                         f"{ts['n_frames']:5d} frames, "
                         f"{ts['n_states']:2d} states, "
                         f"{ts['n_residues']:3d} residues")

        summary.append("=" * 60)

        return "\n".join(summary)
