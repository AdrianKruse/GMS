import json
import time
from typing import List, Dict, Any, Tuple, Union

from core.actions import Action
from core.events import Event


class ReplayRecorder:
    """Record actions and events for replay."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.entries: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def record_action(self, action: Action) -> None:
        """Record a player action."""
        self.entries.append({
            "type": "action",
            "timestamp": time.time() - self.start_time,
            "action_type": action.__class__.__name__,
            "data": action.__dict__
        })
    
    def record_event(self, event: Event) -> None:
        """Record a game event."""
        self.entries.append({
            "type": "event",
            "timestamp": time.time() - self.start_time,
            "event_type": event.__class__.__name__,
            "data": event.__dict__
        })
    
    def save(self) -> bool:
        """Save the recorded replay to a file."""
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.entries, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving replay: {e}")
            return False


class ReplayPlayer:
    """Play back recorded actions and events."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.entries: List[Dict[str, Any]] = []
        self.current_index = 0
        self.start_time = 0
    
    def load(self) -> bool:
        """Load a replay file."""
        try:
            with open(self.filename, 'r') as f:
                self.entries = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading replay: {e}")
            return False
    
    def start(self) -> None:
        """Start replay playback."""
        self.start_time = time.time()
        self.current_index = 0
    
    def get_next_entry(self) -> Tuple[float, str, Dict[str, Any]]:
        """
        Get the next entry to replay.
        
        Returns:
            tuple: (timestamp, entry_type, entry_data)
        """
        if self.current_index >= len(self.entries):
            return -1, "", {}
        
        entry = self.entries[self.current_index]
        self.current_index += 1
        
        return (
            entry["timestamp"],
            entry["type"],
            {"type": entry.get("action_type") or entry.get("event_type"), "data": entry["data"]}
        ) 