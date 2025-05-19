import json
import os
from typing import Dict, List, Optional
from .map_data import GameMap
import logging

logger = logging.getLogger('game')


class MapLoader:
    """Loads and caches map data from JSON files"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MapLoader, cls).__new__(cls)
            cls._instance._maps = {}
            cls._instance._load_all_maps()
        return cls._instance
    
    def _load_all_maps(self) -> None:
        """Load all map JSON files from the maps directory"""
        maps_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(maps_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(maps_dir, filename)
                    with open(filepath, 'r') as f:
                        map_data = json.load(f)
                        # We expect maps to have a name, layout array and entity_mappings
                        if 'name' in map_data and 'layout' in map_data and 'entity_mappings' in map_data:
                            game_map = GameMap(
                                name=map_data['name'],
                                layout=map_data['layout'],
                                entity_mappings=map_data['entity_mappings']
                            )
                            self._maps[game_map.name] = game_map
                except Exception as e:
                    logger.error(f"Error loading map {filename}: {e}")
    
    def get_map(self, map_name: str) -> Optional[GameMap]:
        """Get a map by name"""
        return self._maps.get(map_name)
    
    def get_default_map(self) -> Optional[GameMap]:
        """Get the default map"""

        if not self._maps:
            logger.debug("why is there no map")
            return None

        # Try to get a map named 'default', otherwise return the first map
        return self._maps.get('default') or next(iter(self._maps.values()))
    
    def get_all_map_names(self) -> List[str]:
        """Get a list of all available map names"""
        return list(self._maps.keys())


# Singleton instance for easy access
_loader = None

def get_map_loader() -> MapLoader:
    """Get the singleton map loader instance"""
    global _loader
    if _loader is None:
        _loader = MapLoader()
    return _loader 
