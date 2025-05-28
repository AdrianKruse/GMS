# agents/simple_agent_new.py
"""
Simple agent that still uses the old tower-chasing logic, but can
consume the *new* observation produced by `get_observation_from_round_state`.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from core.round_logic.actions import (
    MOVE_ACTION, ATTACK_ACTION, STAND_ACTION, RESUME_ACTION
)

logger = logging.getLogger("agent")


class SimpleAgent:
    """
    - Decodes the new observation dict (grid_map + vector_state …).
    - Runs the *unchanged* "find-nearest-tower → move / attack / resume" logic.
    """

    # ────────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.pos: Tuple[int, int] = (0, 0)
        self.target_tower: Optional[int] = None

    # ────────────────────────────────────────────────────────────────────
    # Public API ─ exactly the same signature you used before
    # ────────────────────────────────────────────────────────────────────
    def act(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        `new_obs` is the dict returned by `get_observation_from_round_state`.
        """
        legacy_obs = self._decode_observation(new_obs)
        #logger.info(f"Legacy obs: {legacy_obs}")

        # ---------- below this line the *original* behaviour -----------
        self.pos = legacy_obs["position"]

        # continue a running plan?
        if legacy_obs.get("current_active_directive") is not None:
            logger.debug("Continuing existing directive")
            return {"action_type": RESUME_ACTION}

        nearest = self._find_nearest_tower(legacy_obs["towers"])
        if nearest:
            if self._is_adjacent_to(self.pos, nearest["position"]):
                logger.info(f"Attacking tower {nearest['id']}")
                self.target_tower = nearest["id"]
                return {"action_type": ATTACK_ACTION, "target_id": nearest["id"]}

            # choose an adjacent walkable tile
            adj_candidates = self._find_valid_adjacent_positions(
                nearest["position"], legacy_obs["grid_state"]
            )
            if adj_candidates:
                self.target_tower = nearest["id"]
                logger.info(f"Moving to tower {nearest['id']}")
                return {
                    "action_type": MOVE_ACTION,
                    "target_position": adj_candidates[0],
                }
        logger.info(f"Standing still")
        return {"action_type": STAND_ACTION}

    def reset(self) -> None:
        self.target_tower = None

    # ────────────────────────────────────────────────────────────────────
    # NEW: decoder from *modern* obs → legacy structure
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _decode_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the modern observation (grid_map, vector_state, nearest_towers)
        back to the legacy dict expected by SimpleAgent – including tower health.
        """
        g  = obs["grid_map"]         # shape (W, H, 6)
        v  = obs["vector_state"]     # shape (13,)
        nt = obs["nearest_towers"]   # shape (20,) (dx,dy,hp_norm,destroyed) * 5

        W, H, _ = g.shape

        # ──────────────────────────────────────────────────────────────
        # 1. agent position  (channel-0 == 1)
        # ──────────────────────────────────────────────────────────────
        ys, xs = np.where(g[:, :, 0] == 1)
        if xs.size == 0:
            raise RuntimeError("agent position channel empty")
        position: Tuple[int, int] = (int(xs[0]), int(ys[0]))

        # ──────────────────────────────────────────────────────────────
        # 2. passability grid  (0 = free, 1 = wall/tower)
        #    grid_map is already (H, W), so no transpose is needed.
        # ──────────────────────────────────────────────────────────────
        grid_state = (g[:, :, 1] > 0.5).astype(np.int8)      # shape (H, W)

        # ──────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────
        tys, txs = np.where(g[:, :, 2] == 1)
        towers: List[Dict[str, Any]] = []
        for tid, (tx, ty) in enumerate(zip(txs, tys)):
            towers.append(
                dict(
                    id            = tid,
                    position      = (int(tx), int(ty)),
                    health        = 100,      # will patch just below
                    is_destroyed  = False,
                )
            )

        # Patch health/destroyed for the 5 nearest towers (if available).
        if towers:
            towers.sort(key=lambda t: abs(t["position"][0] - position[0])
                                + abs(t["position"][1] - position[1]))
            for idx, tw in enumerate(towers[:5]):           # at most 5 stored
                hp_norm   = np.nan_to_num(nt[idx*4 + 2])    # avoid NaN
                destroyed = bool(nt[idx*4 + 3] > 0.5)
                tw["health"]       = int(hp_norm * 100)
                tw["is_destroyed"] = destroyed

        # ──────────────────────────────────────────────────────────────
        # 4. rebuild current active directive  (vector_state[1:8])
        # ──────────────────────────────────────────────────────────────
        bits = v[1:8]        # 7 numbers: 4 flags + 3 payload fields

        current_active_directive = None
        if bits[:4].sum() > 0:
            if bits[0]:                                          # MOVE flag
                current_active_directive = {"action_type": MOVE_ACTION}
            elif bits[1]:                                        # ATTACK flag
                current_active_directive = {
                    "action_type":   ATTACK_ACTION,
                    "target_position": (int(bits[4] * W), int(bits[5] * H)),
                    "target_id":       int(bits[6]),
                }
            elif bits[2]:
                current_active_directive = {"action_type": STAND_ACTION}
            elif bits[3]:
                current_active_directive = {"action_type": RESUME_ACTION}

        # We cannot reliably rebuild "last interrupted directive" (vector_state
        # only stores a code, not the full flag set), and SimpleAgent never uses
        # it – keep it None.
        last_interrupted_directive = None

        # ──────────────────────────────────────────────────────────────
        # 5. assemble the legacy observation dict
        # ──────────────────────────────────────────────────────────────
        legacy: Dict[str, Any] = dict(
            position                   = position,
            grid_state                 = grid_state,          # (H, W) 0/1
            towers                     = towers,              # list[dict]
            current_active_directive   = current_active_directive,
            last_interrupted_directive = last_interrupted_directive,
        )
        return legacy

    # ────────────────────────────────────────────────────────────────────
    # Original helper methods (unchanged)
    # ────────────────────────────────────────────────────────────────────
    def _find_nearest_tower(self, towers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        nearest, dmin = None, float("inf")
        for tw in towers:
            if tw["health"] <= 0:
                continue
            dx = abs(tw["position"][0] - self.pos[0])
            dy = abs(tw["position"][1] - self.pos[1])
            if dx + dy < dmin:
                nearest, dmin = tw, dx + dy
        return nearest

    @staticmethod
    def _is_adjacent_to(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

    @staticmethod
    def _is_position_valid(x: int, y: int, grid_state: np.ndarray) -> bool:
        h, w = grid_state.shape
        if not (0 <= x < w and 0 <= y < h):
            return False
        return grid_state[y, x] == 0

    def _find_valid_adjacent_positions(
        self, tower_pos: Tuple[int, int], grid_state: np.ndarray
    ) -> List[Tuple[int, int]]:
        res = []
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = tower_pos[0] + dx, tower_pos[1] + dy
            if self._is_position_valid(nx, ny, grid_state):
                res.append((nx, ny))
        return res
