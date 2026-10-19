"""Patient mapping used only by the patient-level sensitivity figures."""

from functools import lru_cache
import os
import re

import pandas as pd


@lru_cache(maxsize=1)
def explicit_mapping() -> dict[str, str] | None:
    path = os.environ.get("AXON_IMMUNE_PATIENT_MAP")
    if not path:
        return None
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"unique_region", "patient_id"}
    if not required.issubset(data.columns):
        raise ValueError("Patient map must contain unique_region and patient_id columns.")
    if data[list(required)].eq("").any().any():
        raise ValueError("Patient map contains an empty region or patient ID.")
    if data.groupby("unique_region").patient_id.nunique().gt(1).any():
        raise ValueError("Patient map assigns different patients to the same region.")
    return data.set_index("unique_region").patient_id.to_dict()


def patient_id_from_region(value: object) -> str:
    region = str(value)
    mapping = explicit_mapping()
    if mapping is not None:
        if region not in mapping:
            raise ValueError(f"Region missing from explicit patient map: {region}")
        return mapping[region]
    match = re.search(r"^/Batch\d+/([^_]+_[^_]+)_", region)
    if match is None:
        raise ValueError(f"Cannot parse patient ID from {region!r}; supply --patient-map.")
    return match.group(1)


def validate_patient_groups(data: pd.DataFrame) -> None:
    ids = data.unique_region.map(patient_id_from_region)
    groups = data.assign(patient_id=ids).groupby("patient_id").progression.nunique()
    if groups.gt(1).any():
        raise ValueError("A mapped patient has conflicting progression labels: " + ", ".join(groups.index[groups.gt(1)]))
