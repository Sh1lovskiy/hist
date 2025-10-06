from pathlib import Path

from pathlib import Path
import json

import subprocess
import sys

def _create_slide(tmp_path: Path, name: str) -> None:
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir(exist_ok=True)
    (slide_dir / f"{name}.tif").write_bytes(b"0")

    anno_dir = tmp_path / "annos"
    anno_dir.mkdir(exist_ok=True)
    payload = {"polygons": [{"label": "tumor", "points": [[0, 0], [10, 0], [10, 10]]}]}
    (anno_dir / f"{name}.json").write_text(json.dumps(payload))


def test_end_to_end(tmp_path: Path) -> None:
    _create_slide(tmp_path, "slide_a")
    _create_slide(tmp_path, "slide_b")

    output_root = tmp_path / "out"
    result = subprocess.run([sys.executable, "-m", "hist.cli",
            "tile",
            "--slides",
            str(tmp_path / "slides"),
            "--annos",
            str(tmp_path / "annos"),
            "--out",
            str(output_root),
            "--patch-size",
            "224",
            "--stride",
            "224",
            "--overwrite",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    csv_dir = output_root / "patch_csvs"
    assert (csv_dir / "slide_a.csv").exists()

    features_dir = tmp_path / "features"
    result = subprocess.run([sys.executable, "-m", "hist.cli",
            "extract",
            "--csv-dir",
            str(csv_dir),
            "--output",
            str(features_dir),
            "--encoder",
            "vit_b16",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (features_dir / "slide_a.pt").exists()

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("slide_id,label\nslide_a,0\nslide_b,1\n")

    train_dir = tmp_path / "train_out"
    result = subprocess.run([sys.executable, "-m", "hist.cli",
            "train",
            "--features",
            str(features_dir),
            "--labels-csv",
            str(labels_csv),
            "--output",
            str(train_dir),
            "--epochs",
            "1",
            "--k-folds",
            "2",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (train_dir / "summary.csv").exists()
