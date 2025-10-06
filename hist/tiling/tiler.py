"""Multi-scale tiler producing patch coordinate CSVs with polygon labels."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from hist.io.annotations import SlideAnnotations, load_annotations
from hist.logging import logger
from hist.utils.paths import ensure_exists

try:  # pragma: no cover - optional dependency
    from openslide import OpenSlide  # type: ignore
except ImportError:  # pragma: no cover - handled in code
    OpenSlide = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.geometry import box as _shapely_box

    _HAVE_SHAPELY = True
except ImportError:  # pragma: no cover - fallback path
    _HAVE_SHAPELY = False
    _ShapelyPolygon = None  # type: ignore
    _shapely_box = None  # type: ignore


DEFAULT_CLASSES = ("BG", "AT", "DYS", "LP", "MM")
CLASS_ID_MAP = {"BG": 0, "AT": 1, "DYS": 2, "LP": 3, "MM": 4}
RASTER_RESOLUTION = 32
_WARNED_SHAPELY = False


class OpenSlideUnavailableError(RuntimeError):
    """Raised when OpenSlide bindings are required but missing."""


@dataclass(frozen=True)
class TileRequest:
    slides_dir: Path
    annotations_dir: Path
    output_dir: Path
    levels: Sequence[int] = (0,)
    patch_size_per_level: Sequence[int] | None = None
    patch_size: int | None = None
    scale_per_level: Sequence[float] | None = None
    stride_per_level: Sequence[int] | None = None
    stride: int | None = None
    label_overlap_threshold: float = 0.5
    overwrite: bool = False
    classes: Sequence[str] = DEFAULT_CLASSES
    write_edges: bool = False


@dataclass
class LevelSpec:
    level: int
    patch_size_level: int
    stride_level: int
    downsample: float

    @property
    def patch_size_level0(self) -> int:
        return max(1, int(round(self.patch_size_level * self.downsample)))

    @property
    def stride_level0(self) -> int:
        return max(1, int(round(self.stride_level * self.downsample)))

    @property
    def patch_area_level0(self) -> float:
        return float(self.patch_size_level0 * self.patch_size_level0)


@dataclass
class PatchRow:
    slide_id: str
    level: int
    patch_id: str
    x: int
    y: int
    w: int
    h: int
    downsample: float
    class_id: int
    class_name: str
    frac_major: float
    class_fractions: Dict[str, float]


def _require_openslide() -> bool:
    if OpenSlide is None:  # pragma: no cover - depends on environment
        logger.warning(
            "OpenSlide not available; using annotation-driven geometry fallback. Install 'openslide-python' for precise tiling.",
        )
        return False
    return True


def _validate_classes(classes: Sequence[str]) -> Tuple[str, ...]:
    classes_tuple = tuple(label.upper() for label in classes)
    unknown = [label for label in classes_tuple if label not in CLASS_ID_MAP]
    if unknown:
        raise ValueError(
            "Unsupported class labels requested: %s" % ", ".join(sorted(set(unknown)))
        )
    if "BG" not in classes_tuple:
        classes_tuple = ("BG",) + classes_tuple
    seen: set[str] = set()
    deduped: List[str] = []
    for label in classes_tuple:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return tuple(deduped)


def _resolve_levels(slide, requested_levels: Sequence[int]) -> Tuple[int, ...]:
    available = getattr(slide, "level_count", 0)
    levels: List[int] = []
    for level in requested_levels:
        if level < 0 or level >= available:
            raise ValueError(f"Requested level {level} outside available range (0-{available - 1})")
        levels.append(int(level))
    return tuple(levels)


def _resolve_sizes(
    levels: Sequence[int],
    patch_size_per_level: Sequence[int] | None,
    patch_size: int | None,
    scale_per_level: Sequence[float] | None,
) -> Tuple[int, ...]:
    if patch_size_per_level is not None:
        if len(patch_size_per_level) != len(levels):
            raise ValueError("--patch-size-per-level must match number of levels")
        return tuple(int(size) for size in patch_size_per_level)
    if patch_size is None:
        raise ValueError("Provide --patch-size-per-level or base --patch-size with --scale-per-level")
    if scale_per_level is None:
        scales = [1.0 for _ in levels]
    else:
        if len(scale_per_level) != len(levels):
            raise ValueError("--scale-per-level must match number of levels")
        scales = [float(scale) for scale in scale_per_level]
    sizes: List[int] = []
    for scale in scales:
        sizes.append(max(1, int(round(patch_size * scale))))
    return tuple(sizes)


def _resolve_strides(
    levels: Sequence[int],
    stride_per_level: Sequence[int] | None,
    stride: int | None,
    patch_sizes: Sequence[int],
) -> Tuple[int, ...]:
    if stride_per_level is not None:
        if len(stride_per_level) != len(levels):
            raise ValueError("--stride-per-level must match number of levels")
        return tuple(int(s) for s in stride_per_level)
    if stride is None:
        stride = patch_sizes[0]
    factors = [patch_size / patch_sizes[0] for patch_size in patch_sizes]
    resolved: List[int] = []
    for factor in factors:
        resolved.append(max(1, int(round(stride * factor))))
    return tuple(resolved)


def _prepare_level_specs(slide, request: TileRequest) -> Tuple[Tuple[int, ...], Dict[int, LevelSpec]]:
    levels = _resolve_levels(slide, request.levels)
    patch_sizes = _resolve_sizes(levels, request.patch_size_per_level, request.patch_size, request.scale_per_level)
    strides = _resolve_strides(levels, request.stride_per_level, request.stride, patch_sizes)
    specs: Dict[int, LevelSpec] = {}
    for idx, level in enumerate(levels):
        downsample = float(slide.level_downsamples[level])
        specs[level] = LevelSpec(
            level=level,
            patch_size_level=int(patch_sizes[idx]),
            stride_level=int(strides[idx]),
            downsample=downsample,
        )
    return levels, specs


def _iter_positions(size: int, window: int, stride: int) -> Iterable[int]:
    if window >= size:
        yield 0
        return
    position = 0
    last_emitted = None
    while position + window <= size:
        yield position
        last_emitted = position
        position += stride
        if position == last_emitted:  # guard against zero stride
            position += 1
    tail_start = size - window
    if last_emitted is None or tail_start > last_emitted:
        yield tail_start


class PolygonAdapter:
    """Utility wrapper for polygon operations with optional shapely support."""

    def __init__(self, points: Sequence[Sequence[float]]):
        coords = [(float(x), float(y)) for x, y in points]
        self.points = coords
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self._geom = _ShapelyPolygon(coords) if _HAVE_SHAPELY else None

    def intersects_box(self, box: Tuple[float, float, float, float]) -> bool:
        minx, miny, maxx, maxy = self.bounds
        bx0, by0, bx1, by1 = box
        return not (maxx < bx0 or maxy < by0 or minx > bx1 or miny > by1)

    def intersection_area(self, box: Tuple[float, float, float, float]) -> float:
        if not self.intersects_box(box):
            return 0.0
        if _HAVE_SHAPELY and self._geom is not None:
            patch_poly = _shapely_box(*box)
            return float(self._geom.intersection(patch_poly).area)
        return _approximate_intersection(self.points, box)


def _approximate_intersection(points: Sequence[Tuple[float, float]], box: Tuple[float, float, float, float]) -> float:
    global _WARNED_SHAPELY
    if not _WARNED_SHAPELY:  # pragma: no cover - log once
        logger.warning(
            "Shapely not available, falling back to raster intersection (resolution=%d).",
            RASTER_RESOLUTION,
        )
        _WARNED_SHAPELY = True
    x0, y0, x1, y1 = box
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    if width == 0 or height == 0:
        return 0.0
    area = width * height
    inside = 0
    total = RASTER_RESOLUTION * RASTER_RESOLUTION
    step_x = width / RASTER_RESOLUTION
    step_y = height / RASTER_RESOLUTION
    half_x = step_x / 2.0
    half_y = step_y / 2.0
    for yi in range(RASTER_RESOLUTION):
        cy = y0 + yi * step_y + half_y
        for xi in range(RASTER_RESOLUTION):
            cx = x0 + xi * step_x + half_x
            if _point_in_polygon(cx, cy, points):
                inside += 1
    return area * (inside / total)


def _point_in_polygon(x: float, y: float, points: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _group_polygons(
    annotations: SlideAnnotations | None,
    class_order: Sequence[str],
    slide_id: str,
) -> Dict[str, List[PolygonAdapter]]:
    grouped: Dict[str, List[PolygonAdapter]] = {label: [] for label in class_order}
    if annotations is None:
        return grouped
    unknown_labels: set[str] = set()
    for polygon in annotations.polygons:
        label = polygon.canonical_label
        if label not in grouped:
            unknown_labels.add(label)
            continue
        grouped[label].append(PolygonAdapter(polygon.points))
    if unknown_labels:
        logger.warning(
            "%s: ignored polygons for unsupported classes: %s",
            slide_id,
            ", ".join(sorted(unknown_labels)),
        )
    return grouped


class _PseudoSlide:
    def __init__(self, width: int, height: int, max_level: int):
        self.level_count = max_level + 1
        self.level_downsamples = [float(2 ** level) for level in range(self.level_count)]
        self.level_dimensions = [
            (
                max(1, int(math.ceil(width / downsample))),
                max(1, int(math.ceil(height / downsample))),
            )
            for downsample in self.level_downsamples
        ]

    def close(self) -> None:  # pragma: no cover - simple noop
        return


def _infer_extent(
    annotations: SlideAnnotations | None,
    fallback: int,
) -> Tuple[int, int]:
    if annotations is None or not annotations.polygons:
        return fallback, fallback
    xs: List[float] = []
    ys: List[float] = []
    for polygon in annotations.polygons:
        for x, y in polygon.points:
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        return fallback, fallback
    width = int(math.ceil(max(xs))) + fallback
    height = int(math.ceil(max(ys))) + fallback
    return max(width, fallback), max(height, fallback)


def _create_pseudo_slide(
    request: TileRequest,
    annotations: SlideAnnotations | None,
) -> _PseudoSlide:
    levels = tuple(request.levels) if request.levels else (0,)
    patch_sizes = _resolve_sizes(levels, request.patch_size_per_level, request.patch_size, request.scale_per_level)
    max_patch_level0 = 0
    for idx, level in enumerate(levels):
        downsample = float(2 ** level)
        size0 = max(1, int(round(patch_sizes[idx] * downsample)))
        max_patch_level0 = max(max_patch_level0, size0)
    base = max(max_patch_level0 * 2, 512)
    width, height = _infer_extent(annotations, base)
    max_level = max(levels) if levels else 0
    return _PseudoSlide(width, height, max_level)


def _compute_fractions(
    patch_box: Tuple[float, float, float, float],
    patch_area: float,
    polygons_by_class: Dict[str, List[PolygonAdapter]],
    class_order: Sequence[str],
) -> Dict[str, float]:
    coverage: Dict[str, float] = {label: 0.0 for label in class_order}
    for label, polygons in polygons_by_class.items():
        if not polygons:
            continue
        total = 0.0
        for polygon in polygons:
            total += polygon.intersection_area(patch_box)
        coverage[label] = total
    if "BG" in coverage:
        non_bg = sum(area for label, area in coverage.items() if label != "BG")
        residual = max(patch_area - non_bg, 0.0)
        coverage["BG"] += residual
    fractions: Dict[str, float] = {}
    for label, area in coverage.items():
        if patch_area <= 0:
            fractions[label] = 0.0
        else:
            fractions[label] = max(0.0, min(area / patch_area, 1.0))
    return fractions


def _resolve_majority(
    fractions: Dict[str, float],
    threshold: float,
) -> Tuple[str, float]:
    best_label = max(fractions.keys(), key=lambda label: fractions[label]) if fractions else "BG"
    frac_major = fractions.get(best_label, 0.0)
    if best_label != "BG" and frac_major < threshold:
        frac_major = fractions.get("BG", 0.0)
        best_label = "BG"
    return best_label, frac_major


def _write_patch_csv(
    path: Path,
    patches: Sequence[PatchRow],
    class_order: Sequence[str],
) -> None:
    header = [
        "slide_id",
        "level",
        "patch_id",
        "x",
        "y",
        "w",
        "h",
        "downsample",
        "class_id",
        "class_name",
        "frac_major",
    ]
    header.extend([f"frac_{label}" for label in class_order])
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for patch in patches:
            row = [
                patch.slide_id,
                patch.level,
                patch.patch_id,
                patch.x,
                patch.y,
                patch.w,
                patch.h,
                patch.downsample,
                patch.class_id,
                patch.class_name,
                patch.frac_major,
            ]
            for label in class_order:
                row.append(patch.class_fractions.get(label, 0.0))
            writer.writerow(row)


def _write_edges(
    path: Path,
    edges: Sequence[Tuple[str, str, str, int, int]],
) -> None:
    header = ["slide_id", "parent_id", "child_id", "parent_level", "child_level"]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for slide_id, parent_id, child_id, parent_level, child_level in edges:
            writer.writerow([slide_id, parent_id, child_id, parent_level, child_level])


def _build_edges(
    slide_id: str,
    levels: Sequence[int],
    patches_by_level: Dict[int, List[PatchRow]],
) -> List[Tuple[str, str, str, int, int]]:
    edges: List[Tuple[str, str, str, int, int]] = []
    tolerance = 2
    for idx in range(len(levels) - 1):
        parent_level = levels[idx]
        child_level = levels[idx + 1]
        parents = patches_by_level.get(parent_level, [])
        children = patches_by_level.get(child_level, [])
        for parent in parents:
            px0, py0 = parent.x, parent.y
            px1 = px0 + parent.w
            py1 = py0 + parent.h
            for child in children:
                cx0, cy0 = child.x, child.y
                cx1 = cx0 + child.w
                cy1 = cy0 + child.h
                if (
                    cx0 >= px0 - tolerance
                    and cy0 >= py0 - tolerance
                    and cx1 <= px1 + tolerance
                    and cy1 <= py1 + tolerance
                ):
                    edges.append((slide_id, parent.patch_id, child.patch_id, parent_level, child_level))
    return edges


def _generate_patches(
    slide,
    slide_id: str,
    class_order: Sequence[str],
    level_specs: Dict[int, LevelSpec],
    levels: Sequence[int],
    polygons_by_class: Dict[str, List[PolygonAdapter]],
    threshold: float,
) -> Tuple[List[PatchRow], Dict[int, List[PatchRow]]]:
    patches: List[PatchRow] = []
    per_level: Dict[int, List[PatchRow]] = {level: [] for level in levels}
    counters: Dict[int, int] = {level: 0 for level in levels}
    for level in levels:
        spec = level_specs[level]
        width, height = slide.level_dimensions[level]
        patch_size_level = spec.patch_size_level
        stride_level = max(1, spec.stride_level)
        for y_level in _iter_positions(height, patch_size_level, stride_level):
            for x_level in _iter_positions(width, patch_size_level, stride_level):
                x0 = int(round(x_level * spec.downsample))
                y0 = int(round(y_level * spec.downsample))
                w0 = spec.patch_size_level0
                h0 = spec.patch_size_level0
                patch_box = (x0, y0, x0 + w0, y0 + h0)
                fractions = _compute_fractions(patch_box, spec.patch_area_level0, polygons_by_class, class_order)
                majority_label, frac_major = _resolve_majority(fractions, threshold)
                class_id = CLASS_ID_MAP.get(majority_label, 0)
                patch_id = f"L{level}_{counters[level]:06d}"
                counters[level] += 1
                row = PatchRow(
                    slide_id=slide_id,
                    level=level,
                    patch_id=patch_id,
                    x=x0,
                    y=y0,
                    w=w0,
                    h=h0,
                    downsample=spec.downsample,
                    class_id=class_id,
                    class_name=majority_label,
                    frac_major=frac_major,
                    class_fractions=fractions,
                )
                patches.append(row)
                per_level[level].append(row)
    return patches, per_level


def tile_dataset(request: TileRequest) -> Path:
    openslide_available = _require_openslide()
    slides_dir = ensure_exists(request.slides_dir, kind="directory", hint="Check --slides path.")
    annotations_dir = ensure_exists(request.annotations_dir, kind="directory", hint="Check --annos path.")

    patch_dir = request.output_dir / "patch_csvs"
    patch_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = request.output_dir / "edges"
    if request.write_edges:
        edges_dir.mkdir(parents=True, exist_ok=True)

    class_order = _validate_classes(request.classes)
    annotations = load_annotations(annotations_dir)
    csv_paths: List[Path] = []

    for slide_path in sorted(slides_dir.iterdir()):
        if slide_path.is_dir():
            continue
        slide_id = slide_path.stem
        csv_path = patch_dir / f"{slide_id}.csv"
        if csv_path.exists() and not request.overwrite:
            logger.info("Skipping %s (exists)", csv_path)
            csv_paths.append(csv_path)
            continue
        slide_annotations = annotations.get(slide_id)
        if slide_annotations is None:
            logger.warning("No annotations for slide %s", slide_id)
        if openslide_available:
            slide = OpenSlide(str(slide_path))  # type: ignore[arg-type]
        else:
            slide = _create_pseudo_slide(request, slide_annotations)
        try:
            levels, level_specs = _prepare_level_specs(slide, request)
            polygons_by_class = _group_polygons(slide_annotations, class_order, slide_id)
            patches, per_level = _generate_patches(
                slide,
                slide_id=slide_id,
                class_order=class_order,
                level_specs=level_specs,
                levels=levels,
                polygons_by_class=polygons_by_class,
                threshold=request.label_overlap_threshold,
            )
            _write_patch_csv(csv_path, patches, class_order)
            logger.info("Created %s", csv_path)
            if request.write_edges and len(levels) > 1:
                edges_path = edges_dir / f"{slide_id}.csv"
                edges = _build_edges(slide_id, levels, per_level)
                _write_edges(edges_path, edges)
                logger.info("Created %s", edges_path)
            csv_paths.append(csv_path)
        finally:
            slide.close()

    logger.info("Generated %d patch CSVs", len(csv_paths))
    return patch_dir


__all__ = ["TileRequest", "tile_dataset", "DEFAULT_CLASSES", "CLASS_ID_MAP"]

