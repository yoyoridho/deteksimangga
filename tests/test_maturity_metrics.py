import pytest
from evaluation.metrics import count_metrics, detection_metrics
from maturity.assessment import assess, harvest_summary
from modules.schema import Observation
from modules.inputs import field_records, projections_from_json, spatial_records


def test_color_never_makes_fruit_harvest_ready():
    o = Observation("a", "v", [0, 0, 10, 10], .9, unique_id="M0001", yellow_fraction=.9)
    fruit = assess([o])[0]
    assert fruit["maturity"] == "Ripe" and fruit["harvest_ready"] is None
    fields = {"M0001": {"maturity": "Mature", "days_after_flowering": 110, "brix": 14, "firmness_n": 20}}
    calibration = {"variety": "TEST ONLY", "validated": True, "min_days": 100, "min_brix": 12, "max_firmness": 30}
    assert assess([o], fields)[0]["harvest_ready"] is None
    assert assess([o], fields, calibration)[0]["harvest_ready"] is True
    fields["M0001"]["brix"] = 8
    assert assess([o], fields, calibration)[0]["harvest_ready"] is False
    del fields["M0001"]["brix"]
    assert assess([o], fields, calibration)[0]["harvest_ready"] is None
    summary = harvest_summary(assess([o]), 400)
    assert summary["unassessed"] == 1 and summary["not_ready"] == 0 and summary["estimated_total_kg"] == .4


def test_invalid_fields_and_unknown_ids():
    o = Observation("a", "v", [0, 0, 10, 10], .9, unique_id="M0001")
    with pytest.raises(ValueError):
        assess([o], {"M0002": {}})
    with pytest.raises(ValueError):
        assess([o], {"M0001": {"brix": "nan"}})
    with pytest.raises(ValueError):
        field_records(b"unique_id,brix\nM0001,2\nM0001,4")
    with pytest.raises(ValueError):
        projections_from_json(b'{"V1": [[1,2]]}')
    with pytest.raises(ValueError):
        spatial_records(b"observation_id,x,y,z\na,nan,0,0", [o])


def test_count_metrics_zero_actual_and_errors():
    result = count_metrics([0, 10, 20], [2, 8, 24])
    assert result["MAE"] == pytest.approx(8/3)
    assert result["RMSE"] == pytest.approx(8**.5)
    assert result["MAPE"] == pytest.approx(20)
    assert result["Counting Error"] == 4 and result["MAPE_excluded_zero_actual"] == 1
    assert count_metrics([0], [0])["MAPE"] is None
    for actual, predicted in [([], []), ([1], [1, 2]), ([1.2], [1]), ([float("nan")], [1])]:
        with pytest.raises(ValueError):
            count_metrics(actual, predicted)


def test_detection_matching_penalizes_duplicate_predictions():
    preds = [{"view_id": "v", "box": [0, 0, 10, 10], "confidence": .9},
             {"view_id": "v", "box": [0, 0, 10, 10], "confidence": .8}]
    result = detection_metrics(preds, {"v": [[0, 0, 10, 10]]})
    assert result["TP"] == 1 and result["FP"] == 1 and result["FN"] == 0
    assert result["Precision"] == .5 and result["Recall"] == 1 and result["F1"] == pytest.approx(2/3)
    assert result["mAP@50"] == 1
    assert detection_metrics([], {"v": [[0, 0, 10, 10]]})["mAP@50"] == 0
    assert detection_metrics([], {"v": []})["mAP@50"] is None
    with pytest.raises(ValueError):
        detection_metrics(preds, {"different": []})
