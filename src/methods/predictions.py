from __future__ import annotations

from pathlib import Path

from src.experiments.results import write_json


def _as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def save_prediction_file(
    path: str | Path,
    *,
    records: list[dict],
    prediction_output,
    id2label: dict[int, str],
):
    logits = _as_list(prediction_output.predictions)
    labels = _as_list(prediction_output.label_ids)
    if len(records) != len(logits) or len(records) != len(labels):
        raise ValueError(
            "Prediction output length does not match source records: "
            f"records={len(records)}, logits={len(logits)}, labels={len(labels)}"
        )

    predictions = []
    for record, logit_values, label_id in zip(records, logits, labels):
        logit_list = [float(value) for value in _as_list(logit_values)]
        predicted_label = int(_argmax(logit_list))
        gold_label = int(label_id)
        predictions.append(
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "label": gold_label,
                "label_name": id2label.get(gold_label),
                "predicted_label": predicted_label,
                "predicted_label_name": id2label.get(predicted_label),
                "logits": logit_list,
            }
        )

    return write_json(
        path,
        {
            "count": len(predictions),
            "predictions": predictions,
        },
    )
