"""Quantitative evaluation framework for dose prediction accuracy.

Evaluates predicted dose distributions using dose-volume histogram (DVH) metrics,
voxel-wise dose error, conformity index, and homogeneity index. Standard organs-at-risk
include spinal cord, brainstem, parotid glands, and mandible.
"""

import numpy as np
import torch


TARGET_DVH_METRICS = ["D1", "D2", "D95", "D98", "D99", "Dmedian"]
OAR_DVH_METRICS = ["D_0.1_cc", "mean"]


def _dvh_metric(roi_dose: np.ndarray, metric: str, num_voxels_within_01_cc: int) -> float:
    """Calculate a single dose-volume histogram metric for a region of interest.

    Args:
        roi_dose: Dose values (Gy) within the ROI.
        metric: DVH metric identifier (D1, D2, D95, D98, D99, Dmedian, D_0.1_cc, mean).
        num_voxels_within_01_cc: Voxel count equivalent to 0.1 cc volume.

    Returns:
        Metric value in Gy.
    """
    if metric == "D1":
        return float(np.percentile(roi_dose, 99))
    elif metric == "D2":
        return float(np.percentile(roi_dose, 98))
    elif metric == "D95":
        return float(np.percentile(roi_dose, 5))
    elif metric == "D98":
        return float(np.percentile(roi_dose, 2))
    elif metric == "D99":
        return float(np.percentile(roi_dose, 1))
    elif metric == "Dmedian":
        return float(np.median(roi_dose))
    elif metric == "D_0.1_cc":
        pct = 100.0 - num_voxels_within_01_cc / len(roi_dose) * 100.0
        return float(np.percentile(roi_dose, pct))
    elif metric == "mean":
        return float(roi_dose.mean())
    raise ValueError(f"Unknown metric: {metric}")


def _dvh_metrics_for_dose(
    dose: np.ndarray,
    ptv_masks: dict[str, torch.Tensor],
    oar_masks: dict[str, torch.Tensor],
    num_voxels_within_01_cc: int,
) -> tuple[dict, dict]:
    """Compute DVH metrics for all planning target volumes and organs-at-risk.

    Args:
        dose: 3D dose distribution (Gy).
        ptv_masks: Binary masks for each PTV structure.
        oar_masks: Binary masks for each OAR structure.
        num_voxels_within_01_cc: Voxel count equivalent to 0.1 cc volume.

    Returns:
        PTV and OAR metric dictionaries with (structure_name, metric) keys.
    """
    ptv_metrics, oar_metrics = {}, {}

    for ptv_name, ptv_tensor in ptv_masks.items():
        ptv_mask = ptv_tensor.numpy().astype(bool)
        for metric in TARGET_DVH_METRICS:
            ptv_metrics[(ptv_name, metric)] = _dvh_metric(dose[ptv_mask], metric, num_voxels_within_01_cc)

    for oar_name, oar_tensor in oar_masks.items():
        oar_mask = oar_tensor.numpy().astype(bool)
        if not oar_mask.any():
            continue
        for metric in OAR_DVH_METRICS:
            oar_metrics[(oar_name, metric)] = _dvh_metric(dose[oar_mask], metric, num_voxels_within_01_cc)

    return ptv_metrics, oar_metrics


def _format_dvh_metrics(
    ptv_metrics: dict,
    oar_metrics: dict,
    ptv_masks: dict[str, torch.Tensor],
    oar_masks: dict[str, torch.Tensor],
    suffix: str,
) -> dict:
    """Format DVH metrics into a tabular structure with standardized column names.

    Args:
        ptv_metrics: PTV metrics with (structure, metric) keys.
        oar_metrics: OAR metrics with (structure, metric) keys.
        ptv_masks: Binary masks for each PTV structure.
        oar_masks: Binary masks for each OAR structure.
        suffix: Label suffix ('pred' or 'target').

    Returns:
        Metrics dictionary with keys: {structure}_{metric}_{suffix}.
    """
    def col(base: str) -> str:
        return f"{base}_{suffix}" if suffix else base

    out = {}
    for ptv_name in ptv_masks:
        for m in ["D99", "D98", "D2", "D1", "Dmedian"]:
            out[col(f"{ptv_name}_{m}")] = ptv_metrics[(ptv_name, m)]
    for oar_name in oar_masks:
        for m in OAR_DVH_METRICS:
            key = (oar_name, m)
            if key in oar_metrics:
                label = m.replace(".", "").replace(" ", "_")
                out[col(f"{oar_name}_{label}")] = oar_metrics[key]
    return out

def Nakamura_CI(
    ptv_vol: torch.Tensor, 
    dose_vol: torch.Tensor, 
    Rx: float,
)-> float: 
    """Calculate Nakamura Conformity Index (nCI) for dose conformity assessment.

    Evaluates the conformality of the 95% prescription isodose surface to the PTV.
    All volumes must be co-registered on the same spatial grid.

    Args:
        ptv_vol: Binary PTV mask of shape (D, H, W).
        dose_vol: 3D dose distribution (Gy) of shape (D, H, W).
        Rx: Prescription dose (Gy).

    Returns:
        nCI value (unitless, optimal value = 1.0).
    """
    # Compute the ptv volume (in voxels)
    ptv_volume = ptv_vol.sum().item()

    # Compute the volume of the dose that meets or exceeds the prescription dose (Rx)
    dose_volume = (dose_vol >= 0.95 * Rx).sum().item()
    # Compute the intersection volume of the PTV and the dose volume
    intersection_volume = ((dose_vol >= 0.95 * Rx) & (ptv_vol == 1)).sum().item()

    # Compute the Nakamura Conformity
    if intersection_volume == 0:
        raise ValueError("Intersection volume must be greater than zero to compute conformity index.")
    
    nCI = (dose_volume * ptv_volume) / (intersection_volume ** 2) 
    return nCI

def compute_patient_metrics(
    pred_dose: torch.Tensor,
    target_dose: torch.Tensor,
    possible_dose_mask: torch.Tensor,
    ptv_masks: dict[str, torch.Tensor],
    oar_masks: dict[str, torch.Tensor],
    voxel_size: float,
    Rx: float | None = None, 
) -> dict:
    """Compute comprehensive dosimetric evaluation metrics for a single patient.

    Calculates dose score (mean absolute error), DVH score (mean DVH metric deviation),
    homogeneity index, and Nakamura conformity index and standard DVH metrics. All inputs 
    must be co-registered.

    Args:
        pred_dose: Predicted dose distribution (Gy).
        target_dose: Reference dose distribution (Gy).
        possible_dose_mask: Evaluation region mask (typically body contour).
        ptv_masks: PTV structure masks. First entry is treated as the primary high-dose PTV.
        oar_masks: OAR structure masks (e.g., {"SpinalCord": ..., "Brainstem": ...}).
        voxel_size: Voxel volume (mm³) for D_0.1cc calculation.
        Rx: Prescription dose (Gy) for conformity index. Required if conformity is computed.

    Returns:
        Metrics dictionary containing dose_score, dvh_score, homogeneity_score, nCI, and DVH metrics.
    """
    pred = pred_dose.numpy()
    target = target_dose.numpy()
    mask = possible_dose_mask.numpy().astype(bool)

    num_voxels_within_01_cc = int(np.maximum(1, np.round(100.0 / voxel_size)))

    # Dose score: MAE over the possible dose region
    dose_score = float(np.mean(np.abs(pred[mask] - target[mask])))

    pred_ptv, pred_oar = _dvh_metrics_for_dose(pred, ptv_masks, oar_masks, num_voxels_within_01_cc)
    target_ptv, target_oar = _dvh_metrics_for_dose(target, ptv_masks, oar_masks, num_voxels_within_01_cc)

    # DVH score: mean absolute difference across all PTV and OAR metrics
    all_diffs = (
        [abs(pred_ptv[k] - target_ptv[k]) for k in pred_ptv]
        + [abs(pred_oar[k] - target_oar[k]) for k in pred_oar]
    )
    dvh_score = float(np.mean(all_diffs))

    # Homogeneity index for the first PTV (pred and target)
    high_dose_ptv = next(iter(ptv_masks))
    hi_pred = ((pred_ptv[(high_dose_ptv, "D2")] - pred_ptv[(high_dose_ptv, "D98")]) / pred_ptv[(high_dose_ptv, "Dmedian")]) * 100.0
    hi_target = ((target_ptv[(high_dose_ptv, "D2")] - target_ptv[(high_dose_ptv, "D98")]) / target_ptv[(high_dose_ptv, "Dmedian")]) * 100.0

    # Compute the Nakamura Conformity Index for the input and predicted dose
    if Rx is not None : 
        target_nCI = Nakamura_CI(ptv_masks[high_dose_ptv], target_dose, Rx)
        pred_nCI = Nakamura_CI(ptv_masks[high_dose_ptv], pred_dose, Rx)
    else : 
        target_nCI = None
        pred_nCI = None

    return {
        "dose_score": dose_score,
        "dvh_score": dvh_score,
        f"homogeneity_score_{high_dose_ptv}_pred": hi_pred,
        f"homogeneity_score_{high_dose_ptv}_target": hi_target,
        f"{high_dose_ptv}_Nakamura_CI_pred": pred_nCI,
        f"{high_dose_ptv}_Nakamura_CI_target": target_nCI,
        **_format_dvh_metrics(pred_ptv, pred_oar, ptv_masks, oar_masks, suffix="pred"),
        **_format_dvh_metrics(target_ptv, target_oar, ptv_masks, oar_masks, suffix="target"),
    }
