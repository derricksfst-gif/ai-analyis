"""Kokumi inference with final_model.joblib."""

import importlib.util
import sys
import types
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "final_model.joblib"


def _alias_numpy_core():
    try:
        import numpy.core as core

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)
        sys.modules.setdefault("numpy._core._multiarray_umath", core._multiarray_umath)
    except Exception:
        pass


def _register_predictor_module():
    if "src.final_pipeline" in sys.modules:
        return
    root = Path(__file__).resolve().parent
    source = root / "src" / "final_pipeline.py"
    if not source.exists():
        source = root / "final_pipeline.py"
    package = types.ModuleType("src")
    package.__path__ = [str(source.parent)]
    sys.modules["src"] = package
    spec = importlib.util.spec_from_file_location("src.final_pipeline", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.final_pipeline"] = module
    spec.loader.exec_module(module)


def load_model(model_path=MODEL_PATH):
    _alias_numpy_core()
    _register_predictor_module()
    return joblib.load(model_path)


def predict_frame(model, frame):
    """frame columns: name, smiles. Returns the predictor table."""
    return model.predict_dataframe(frame)
