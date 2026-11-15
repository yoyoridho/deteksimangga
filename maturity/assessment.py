from collections import defaultdict
import numpy as np
from modules.schema import finite_number


MATURITY = ("Unripe", "Mature", "Ripe")


def assess(observations, fields=None, calibration=None):
    fields = fields or {}
    groups = defaultdict(list)
    for obs in observations:
        groups[obs.unique_id].append(obs)
    fruits = []
    if calibration:
        if not str(calibration.get("variety", "")).strip() or not calibration.get("validated"):
            raise ValueError("Kalibrasi harus memiliki varietas dan konfirmasi validasi lapangan.")
        thresholds = {key: finite_number(calibration.get(key), key) for key in ("min_days", "min_brix", "max_firmness")}
    for uid, items in groups.items():
        record = fields.get(uid, {})
        yellow = float(np.median([item.yellow_fraction for item in items]))
        visual = "Ripe" if yellow > 0.65 else "Mature" if yellow > 0.25 else "Unripe"
        measured_class = record.get("maturity")
        if measured_class and measured_class not in MATURITY:
            raise ValueError(f"Kelas kematangan {uid} harus Unripe, Mature, atau Ripe.")
        maturity = measured_class or visual
        ready = None
        measurements = {}
        for key in ("days_after_flowering", "brix", "firmness_n", "weight_g"):
            if record.get(key) not in (None, ""):
                measurements[key] = finite_number(record[key], f"{uid}.{key}")
        if calibration and measured_class and all(key in measurements for key in ("days_after_flowering", "brix", "firmness_n")):
            ready = bool(maturity in {"Mature", "Ripe"} and measurements["days_after_flowering"] >= thresholds["min_days"]
                         and measurements["brix"] >= thresholds["min_brix"] and measurements["firmness_n"] <= thresholds["max_firmness"])
        fruits.append({
            "unique_id": uid, "observations": len(items), "maturity": maturity,
            "maturity_basis": "field" if measured_class else "visual-proxy-unvalidated",
            "yellow_fraction": yellow, "harvest_ready": ready,
            "variety": calibration["variety"] if calibration else None, **measurements,
        })
    unknown_ids = set(fields) - set(groups)
    if unknown_ids:
        raise ValueError(f"Data lapangan memiliki ID yang tidak tersedia: {', '.join(sorted(unknown_ids))}")
    return fruits


def harvest_summary(fruits, average_weight_g=None):
    total = len(fruits)
    ready = sum(fruit["harvest_ready"] is True for fruit in fruits)
    not_ready = sum(fruit["harvest_ready"] is False for fruit in fruits)
    if average_weight_g is not None:
        average_weight_g = finite_number(average_weight_g, "Bobot rata-rata", minimum=0.001)
    return {"unique": total, "ready": ready, "not_ready": not_ready, "unassessed": total-ready-not_ready,
            "readiness_percent": 100*ready/total if total else None,
            "estimated_total_kg": total*average_weight_g/1000 if average_weight_g else None,
            "estimated_ready_kg": ready*average_weight_g/1000 if average_weight_g else None}
