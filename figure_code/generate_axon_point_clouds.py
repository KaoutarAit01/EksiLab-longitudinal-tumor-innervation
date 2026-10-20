"""Sample interior pixel coordinates from axon segmentation masks."""

from pathlib import Path
import argparse
import csv
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


AXONS_PATH = Path("data/RP_Neuronal_data_axons.h5ad")
# AXONS_PATH = Path("data/AS_Neuronal_data_axons.h5ad")
MASK_TABLE = Path("data/mask_paths.csv")
OUTPUT_DIR = Path("axon_point_clouds")


def read_observations(path):
    path = Path(path)
    if path.suffix.lower() == ".h5ad":
        data = ad.read_h5ad(path, backed="r")
        try:
            frame = data.obs.copy()
            frame.index = frame.index.astype(str)
        finally:
            data.file.close()
        return frame
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError("Input must be an H5AD or CSV file")


def coordinates(frame, x_col=None, y_col=None):
    def choose(explicit, candidates):
        if explicit:
            if explicit not in frame:
                raise ValueError(f"Missing coordinate column: {explicit}")
            return explicit
        for name in candidates:
            if name in frame:
                return name
        raise ValueError(f"Missing coordinate column; expected one of {candidates}")
    x_col = choose(x_col, ["x", "centroid_x", "centroid-1"])
    y_col = choose(y_col, ["y", "centroid_y", "centroid-0"])
    return frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def group_keys(frame, columns):
    """Match masks within the same sample or scene."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing grouping columns: {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError("Grouping columns contain missing values")
    clean = frame[columns].astype(str).apply(lambda s: s.str.strip())
    if clean.eq("").any().any():
        raise ValueError("Grouping columns contain empty values")
    return list(clean.itertuples(index=False, name=None))


def load_mask(path, kind, allow_large):
    if allow_large:
        Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Expected a single 2D mask, not a multipage image: {path}")
        mask = np.array(image)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale or label mask: {path}")
    if not np.isfinite(mask).all() or (mask < 0).any():
        raise ValueError(f"Mask contains invalid values: {path}")
    if kind == "binary":
        labeled, count = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=int))
        original_ids = np.arange(count + 1)
    else:
        if not np.equal(mask, np.floor(mask)).all():
            raise ValueError(f"Instance labels must be nonnegative integers: {path}")
        # Zero is always background.
        original_ids = np.concatenate(([0], np.unique(mask[mask > 0])))
        labeled = np.searchsorted(original_ids, mask).astype(np.int32)
        count = len(original_ids) - 1
    if count == 0:
        return labeled, original_ids, np.empty((0, 2)), []
    centers_yx = ndimage.center_of_mass(labeled > 0, labeled, np.arange(1, count + 1))
    centers_xy = np.asarray(centers_yx)[:, ::-1]
    return labeled, original_ids, centers_xy, ndimage.find_objects(labeled)


def sample_region(labeled, slices, region_id, n_points):
    """Take evenly spaced pixels in row order, returned as (x, y)."""
    ys, xs = slices[region_id - 1]
    yx = np.argwhere(labeled[ys, xs] == region_id)
    yx += np.array([ys.start, xs.start])
    if len(yx) > n_points:
        yx = yx[np.linspace(0, len(yx) - 1, n_points, dtype=int)]
    return yx[:, ::-1]


def run(args):
    if args.points < 1:
        raise ValueError("--points must be positive")
    if args.max_match_distance is not None and args.max_match_distance < 0:
        raise ValueError("--max-match-distance must be nonnegative")
    if args.cell_distance < 0:
        raise ValueError("--cell-distance must be nonnegative")
    if args.match == "label" and args.mask_kind != "labels":
        raise ValueError("Label matching requires --mask-kind labels")
    obs = read_observations(args.axons)
    reserved = {"cloud_axon_id", "cloud_observation_id", "point_index", "x_cloud", "y_cloud"}
    if reserved.intersection(obs.columns):
        raise ValueError(f"Input contains reserved output fields: {sorted(reserved.intersection(obs.columns))}")
    keys = group_keys(obs, args.group_by)
    xy = coordinates(obs, args.x_col, args.y_col)
    manifest = pd.read_csv(args.masks, dtype=str)
    manifest_keys = group_keys(manifest, args.group_by)
    if "mask_path" not in manifest or manifest.mask_path.isna().any():
        raise ValueError("Mask table requires a nonempty mask_path column")
    if len(set(manifest_keys)) != len(manifest_keys):
        raise ValueError("Mask table must contain exactly one row per grouping key")
    paths = {}
    for key, value in zip(manifest_keys, manifest.mask_path):
        path = Path(value)
        paths[key] = path if path.is_absolute() else args.masks.parent / path
    missing = set(keys) - set(paths)
    if missing:
        raise ValueError(f"No mask mapping for groups: {sorted(missing)[:10]}")
    for key in set(keys):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    if args.match == "label" and args.label_col not in obs:
        raise ValueError(f"Missing instance label column: {args.label_col}")

    cell_trees = {}
    if args.cells:
        cells = read_observations(args.cells)
        cell_keys = group_keys(cells, args.group_by)
        cell_xy = coordinates(cells, args.cell_x_col, args.cell_y_col)
        grouped = {}
        for i, key in enumerate(cell_keys):
            if np.isfinite(cell_xy[i]).all():
                grouped.setdefault(key, []).append(cell_xy[i])
        cell_trees = {key: cKDTree(values) for key, values in grouped.items()}

    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    fields = ["cloud_axon_id", "cloud_observation_id", *obs.columns, "point_index", "x_cloud", "y_cloud"]
    point_path = output / "axon_cloud_points.csv"
    with point_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for key, indices in groups.items():
            labeled, original_ids, centers, slices = load_mask(paths[key], args.mask_kind, args.allow_large_masks)
            tree = cKDTree(centers) if len(centers) else None
            id_lookup = {int(value): i for i, value in enumerate(original_ids) if i > 0}
            current = []
            for i in indices:
                rec = dict(cloud_axon_id=i + 1, cloud_observation_id=str(obs.index[i]),
                           **dict(zip(args.group_by, key)), mask_path=str(paths[key]),
                           status="pending", matched_label=None, match_distance=np.nan,
                           nearest_cell_distance=np.nan, n_points=0)
                rec["_row"] = i
                if not np.isfinite(xy[i]).all():
                    rec["status"] = "invalid_coordinates"
                elif args.cells and key not in cell_trees:
                    rec["status"] = "no_cells_in_group"
                elif args.cells:
                    distance = float(cell_trees[key].query(xy[i])[0])
                    rec["nearest_cell_distance"] = distance
                    if distance > args.cell_distance:
                        rec["status"] = "outside_cell_distance"
                if rec["status"] == "pending":
                    if tree is None:
                        rec["status"] = "empty_mask"
                    elif args.match == "label":
                        value = pd.to_numeric(obs.iloc[i][args.label_col], errors="coerce")
                        valid_label = pd.notna(value) and np.isfinite(value) and value == int(value)
                        region_id = id_lookup.get(int(value)) if valid_label else None
                        if region_id is None:
                            rec["status"] = "label_not_found"
                        else:
                            rec["_region"] = region_id
                    else:
                        distance, index = tree.query(xy[i])
                        rec["_region"] = int(index) + 1
                    if "_region" in rec:
                        region_id = rec["_region"]
                        rec["matched_label"] = int(original_ids[region_id])
                        rec["match_distance"] = float(np.linalg.norm(xy[i] - centers[region_id - 1]))
                        if args.max_match_distance is not None and rec["match_distance"] > args.max_match_distance:
                            rec["status"] = "outside_match_distance"
                current.append(rec)
            # Reject duplicate matches to the same region.
            assignments = {}
            for rec in current:
                if rec["status"] == "pending":
                    assignments.setdefault(rec["_region"], []).append(rec)
            for matches in assignments.values():
                if len(matches) > 1:
                    for rec in matches:
                        rec["status"] = "duplicate_region_match"
            for rec in current:
                i = rec.pop("_row")
                region_id = rec.pop("_region", None)
                if rec["status"] == "pending":
                    points = sample_region(labeled, slices, region_id, args.points)
                    metadata = ["" if pd.isna(value) else value for value in obs.iloc[i].tolist()]
                    for j, (x, y) in enumerate(points, 1):
                        writer.writerow([i + 1, str(obs.index[i]), *metadata, j, int(x), int(y)])
                    rec["status"] = "matched"
                    rec["n_points"] = len(points)
                records.append(rec)
    report = pd.DataFrame(records)
    if report.empty:
        raise ValueError("No axons in input")
    report.to_csv(output / "axon_mask_matches.csv", index=False)
    counts = report.status.value_counts().to_dict()
    summary = dict(axons=str(args.axons), mask_table=str(args.masks), group_by=args.group_by,
                   mask_kind=args.mask_kind, matching=args.match, max_match_distance=args.max_match_distance,
                   requested_points=args.points, input_axons=len(obs), status_counts=counts,
                   total_points=int(report.n_points.sum()),
                   cells=str(args.cells) if args.cells else None,
                   cell_distance=args.cell_distance if args.cells else None)
    (output / "point_cloud_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary['total_points']:,} points from {counts.get('matched', 0):,}/{len(obs):,} axons")
    if counts.get("matched", 0) < len(obs):
        warnings.warn("Some axons have no point cloud; inspect axon_mask_matches.csv")
    if not counts.get("matched", 0):
        raise RuntimeError("No axons matched a mask region; inspect the matching report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axons", type=Path, default=AXONS_PATH)
    parser.add_argument("--masks", type=Path, default=MASK_TABLE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--group-by", nargs="+", default=["sample_id"])
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--mask-kind", choices=["binary", "labels"], default="binary")
    parser.add_argument("--match", choices=["nearest", "label"], default="nearest")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-match-distance", type=float, default=None)
    parser.add_argument("--x-col")
    parser.add_argument("--y-col")
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--cell-distance", type=float, default=200)
    parser.add_argument("--cell-x-col")
    parser.add_argument("--cell-y-col")
    parser.add_argument("--allow-large-masks", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
