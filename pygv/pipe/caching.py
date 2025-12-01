import hashlib
from pathlib import Path


class CacheManager:
    """Manages dataset caching"""

    def __init__(self, config):
        self.config = config

    def get_dataset_hash(self):
        """Generate unique hash for dataset based on parameters"""
        # Create hash from relevant parameters
        hash_input = f"{self.config.traj_dir}_{self.config.selection}_{self.config.stride}_{self.config.n_neighbors}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

    def check_cached_dataset(self, dataset_hash):
        """Check if cached dataset exists"""
        if not self.config.cache:
            return None

        cache_dir = Path(self.config.cache_dir) if hasattr(self.config, 'cache_dir') else Path(
            self.config.output_dir) / 'cache'
        cache_file = cache_dir / f"dataset_{dataset_hash}.pkl"

        return str(cache_file) if cache_file.exists() else None

    def cache_dataset(self, dataset_path, dataset_hash):
        """Cache dataset for future use"""
        # Implementation depends on your dataset format
        # TODO: Implement this
        pass