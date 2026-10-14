from pathlib import Path

import numpy as np

from src.phone_face.run import Box, draw_boxes, parse_label


def test_parse_label_accepts_crlf_and_confidence(tmp_path: Path):
    label = tmp_path / "a.txt"
    label.write_bytes(b"0 0.5 0.5 0.2 0.4\r\n1 0.2 0.3 0.1 0.1 0.9\r\n")
    boxes, issues = parse_label(label)
    assert issues == []
    assert [box.class_id for box in boxes] == [0, 1]
    assert boxes[1].confidence == 0.9


def test_parse_label_rejects_invalid_rows(tmp_path: Path):
    label = tmp_path / "bad.txt"
    label.write_text("2 0.5 0.5 0.2 0.2\n0 1.4 0.5 0.2 0.2\n", encoding="utf-8")
    boxes, issues = parse_label(label)
    assert boxes == []
    assert issues == ["line_1:unknown_class_2", "line_2:coordinate_out_of_range"]


def test_parse_label_recovers_concatenated_rows(tmp_path: Path):
    label = tmp_path / "joined.txt"
    label.write_text("0 0.2 0.3 0.1 0.1 0 0.7 0.8 0.2 0.2\n", encoding="utf-8")
    boxes, issues = parse_label(label)
    assert len(boxes) == 2
    assert issues == ["line_1:concatenated_2_rows"]


def test_parse_label_removes_exact_duplicates(tmp_path: Path):
    label = tmp_path / "duplicates.txt"
    label.write_text("1 0.2 0.3 0.1 0.1\n1 0.2 0.3 0.1 0.1\n", encoding="utf-8")
    boxes, issues = parse_label(label)
    assert len(boxes) == 1
    assert issues == ["duplicate_rows_1"]


def test_draw_boxes_changes_pixels():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = draw_boxes(image, [Box(0, 0.5, 0.5, 0.2, 0.2)])
    assert rendered.sum() > 0
