import unittest

from src.data.label_policy import (
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    build_label_from_annotators,
    extract_annotator_label_ids,
    extract_annotator_label_names,
    majority_vote_label_id,
)


class LabelPolicyTests(unittest.TestCase):
    def test_label_mapping_matches_hugging_face_class_label_order(self):
        self.assertEqual(
            LABEL_ID_TO_NAME,
            {0: "hatespeech", 1: "normal", 2: "offensive"},
        )
        self.assertEqual(
            LABEL_NAME_TO_ID,
            {"hatespeech": 0, "normal": 1, "offensive": 2},
        )

    def test_extracts_raw_json_annotator_string_labels(self):
        example = {
            "annotators": [
                {"label": "hatespeech", "annotator_id": 203, "target": ["Hindu"]},
                {"label": "offensive", "annotator_id": 204, "target": ["Hindu"]},
                {"label": "offensive", "annotator_id": 233, "target": ["Other"]},
            ]
        }

        self.assertEqual(extract_annotator_label_ids(example), [0, 2, 2])
        self.assertEqual(
            extract_annotator_label_names(example),
            ["hatespeech", "offensive", "offensive"],
        )

    def test_extracts_hugging_face_sequence_class_label_ids(self):
        example = {
            "annotators": {
                "label": [0, 2, 2],
                "annotator_id": [203, 204, 233],
                "target": [["Hindu"], ["Hindu"], ["Other"]],
            }
        }

        self.assertEqual(extract_annotator_label_ids(example), [0, 2, 2])

    def test_majority_vote_requires_strict_majority(self):
        self.assertEqual(majority_vote_label_id([0, 2, 2]), 2)
        self.assertIsNone(majority_vote_label_id([0, 1, 2]))

    def test_build_label_returns_majority_label_id(self):
        example = {"annotators": {"label": [0, 2, 2]}}

        self.assertEqual(build_label_from_annotators(example), 2)

    def test_build_label_can_return_none_for_no_majority(self):
        example = {"annotators": {"label": [0, 1, 2]}}

        self.assertIsNone(build_label_from_annotators(example, on_no_majority="return_none"))

    def test_build_label_raises_by_default_for_no_majority(self):
        example = {"annotators": {"label": [0, 1, 2]}}

        with self.assertRaises(ValueError):
            build_label_from_annotators(example)


if __name__ == "__main__":
    unittest.main()
