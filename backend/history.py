"""
Experiment history persistence for AutoMiniLLM.

Stores each compression experiment as a JSON record under `experiments/history/`,
supporting list/add/get/delete/clear. The history is the source of truth for the
dashboard's comparison views.
"""
import json
import logging
import os
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "history")
_lock = threading.Lock()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _file_path(exp_id: str) -> str:
    _ensure_dir(HISTORY_DIR)
    return os.path.join(HISTORY_DIR, f"{exp_id}.json")


def add_experiment(result: Dict) -> Dict:
    """Persist an experiment result to JSON history."""
    exp_id = result.get("id") or f"{result.get('timestamp','exp').replace(' ','_')}"
    with _lock:
        _ensure_dir(HISTORY_DIR)
        path = _file_path(exp_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    logger.info("Saved experiment %s to %s", exp_id, path)
    return result


def list_experiments() -> List[Dict]:
    """Return all experiments, newest first."""
    _ensure_dir(HISTORY_DIR)
    exps = []
    for fn in os.listdir(HISTORY_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                exps.append(json.load(f))
        except Exception as e:
            logger.warning("Skipping corrupt history file %s: %s", fn, e)
    exps.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    return exps


def get_experiment(exp_id: str) -> Optional[Dict]:
    path = _file_path(exp_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_experiment(exp_id: str) -> bool:
    path = _file_path(exp_id)
    with _lock:
        if os.path.exists(path):
            os.remove(path)
            return True
    return False


def clear_history() -> int:
    """Delete all experiments, returns count removed."""
    _ensure_dir(HISTORY_DIR)
    count = 0
    with _lock:
        for fn in os.listdir(HISTORY_DIR):
            if fn.endswith(".json"):
                try:
                    os.remove(os.path.join(HISTORY_DIR, fn))
                    count += 1
                except OSError:
                    pass
    return count
