import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "models" / "best_model.joblib"
EXPECTED_FEATURE_COUNT = 328


class MLPredictionError(RuntimeError):
    """Raised when the trained model cannot produce a prediction."""


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, tuple[str, ...]]:
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        feature_names = tuple(artifact["metadata"]["feature_names"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        LOGGER.exception("Unable to load ML model artifact from %s", MODEL_PATH)
        raise MLPredictionError("The ML prediction model is unavailable.") from error
    except Exception as error:
        LOGGER.exception("Unexpected error loading ML model artifact from %s", MODEL_PATH)
        raise MLPredictionError("The ML prediction model is unavailable.") from error

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        LOGGER.error(
            "Expected %s model features but found %s",
            EXPECTED_FEATURE_COUNT,
            len(feature_names),
        )
        raise MLPredictionError("The ML prediction model has invalid feature metadata.")

    return model, feature_names


def predict_symptoms(symptoms: str) -> dict[str, Any]:
    """Predict the top conditions for free-text symptoms."""
    model, feature_names = _load_model()
    normalized_input = _normalize_text(symptoms)
    vector = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.int8)
    recognized_symptoms: list[str] = []

    for index, feature_name in enumerate(feature_names):
        normalized_feature = _normalize_text(feature_name)
        if normalized_feature and re.search(
            rf"(?<!\w){re.escape(normalized_feature)}(?!\w)",
            normalized_input,
        ):
            vector[index] = 1
            recognized_symptoms.append(feature_name)

    if not recognized_symptoms:
        return {
            "predictions": [],
            "recognized_symptoms": [],
            "message": (
                "No recognized symptoms found. Please enter symptoms from the "
                "available symptom vocabulary."
            ),
        }

    try:
        probabilities = model.predict_proba(vector.reshape(1, -1))[0]
        ranked_indices = np.argsort(probabilities)[::-1][:3]
        predictions = [
            {
                "condition": str(model.classes_[index]),
                "model_score": round(float(probabilities[index]) * 100, 2),
            }
            for index in ranked_indices
        ]
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        LOGGER.exception("ML prediction failed")
        raise MLPredictionError("The ML prediction could not be completed.") from error
    except Exception as error:
        LOGGER.exception("Unexpected error during ML prediction")
        raise MLPredictionError("The ML prediction could not be completed.") from error

    return {
        "predictions": predictions,
        "recognized_symptoms": recognized_symptoms,
        "model_scores_label": "Model scores (%)",
    }
