from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app():
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()


def test_navigation_without_data(app):
    assert not app.exception
    for page in ["Akuisisi & Analisis", "Asosiasi & ID", "Kematangan & Panen", "Evaluasi", "Riwayat & Ekspor", "Dataset & Model"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_demo_dashboard_all_data_pages(app):
    next(button for button in app.button if button.label == "Coba data uji sintetis").click().run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:4]] == ["3", "9", "3", "6"]
    for page in ["Asosiasi & ID", "Kematangan & Panen", "Evaluasi", "Riwayat & Ekspor"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_demo_video_accessible_without_mango_model(app):
    app.sidebar.selectbox[0].set_value("Sintetis · data uji").run()
    app.sidebar.radio[0].set_value("Akuisisi & Analisis").run()
    next(button for button in app.button if button.label == "Demo video sintetis · ByteTrack").click().run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Dashboard").run()
    assert not app.exception
    assert [metric.value for metric in app.metric[:4]] == ["24", "72", "3", "69"]
