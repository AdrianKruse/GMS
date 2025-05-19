import json
import os
from typing import Dict, List, Optional


class SpriteLoader:
    """Loads and caches sprite data from JSON files"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpriteLoader, cls).__new__(cls)
            cls._instance._sprites = {}
            cls._instance._load_all_sprites()
        return cls._instance
    
    def _load_all_sprites(self) -> None:
        """Load all sprite JSON files from the sprites directory"""
        sprites_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(sprites_dir):
            if filename.endswith('.json'):
                sprite_name = filename[:-5] # Remove .json extension
                try:
                    filepath = os.path.join(sprites_dir, filename)
                    with open(filepath, 'r') as f:
                        sprite_data = json.load(f)
                        # The actual sprite data is under the "sprite" key in the JSON
                        if "sprite" in sprite_data and isinstance(sprite_data["sprite"], list):
                            self._sprites[sprite_name] = sprite_data["sprite"]
                        else:
                            print(f"Warning: Sprite data not found or invalid format in {filename}")
                except Exception as e:
                    print(f"Error loading sprite {filename}: {e}")
        # Log all loaded sprite names
        print(f"Loaded sprites: {list(self._sprites.keys())}")
    
    def get_sprite(self, sprite_name: str) -> Optional[List[str]]:
        """Get a sprite by name"""
        return self._sprites.get(sprite_name)
    
    def get_sprite_width(self, sprite_name: str) -> int:
        """Get the width of a sprite"""
        sprite = self.get_sprite(sprite_name)
        if not sprite:
            return 5  # Default width
        return len(sprite[0])
    
    def get_sprite_height(self, sprite_name: str) -> int:
        """Get the height of a sprite"""
        sprite = self.get_sprite(sprite_name)
        if not sprite:
            return 5  # Default height
        return len(sprite)
    
    def get_all_sprite_names(self) -> List[str]:
        """Get a list of all available sprite names"""
        return list(self._sprites.keys())


# Singleton instance for easy access
_loader = None

def get_loader() -> SpriteLoader:
    """Get the singleton sprite loader instance"""
    global _loader
    if _loader is None:
        _loader = SpriteLoader()
    return _loader 