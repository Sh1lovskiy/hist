from pathlib import Path

import json

from hist.io.annotations import BACKGROUND_ALIASES, parse_annotation_file


def test_parse_dict(tmp_path: Path) -> None:
    payload = {
        "polygons": [
            {"label": "Tumor", "points": [[0, 0], [1, 0], [1, 1]]},
            {"label": "BG", "points": [[2, 2], [3, 2], [3, 3]]},
        ]
    }
    path = tmp_path / "slide.json"
    path.write_text(json.dumps(payload))
    ann = parse_annotation_file(path)
    assert ann.slide_id == "slide"
    assert len(ann.polygons) == 2
    assert any(poly.label == "background" for poly in ann.polygons)


def test_parse_list(tmp_path: Path) -> None:
    payload = [
        {"label": "background", "points": [[0, 0], [1, 0], [1, 1]]},
        {"label": "tumor", "points": [[2, 2], [3, 2], [3, 3]]},
    ]
    path = tmp_path / "slide.json"
    path.write_text(json.dumps(payload))
    ann = parse_annotation_file(path)
    assert ann.slide_id == "slide"
    assert {poly.label for poly in ann.polygons} == {"background", "tumor"}
