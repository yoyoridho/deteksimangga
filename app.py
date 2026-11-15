from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import copy
import hashlib
import json
import logging

import altair as alt
import pandas as pd
import streamlit as st

from dataset.synthetic import generate
from dataset.prepare import prepare
from detection.yolo import Detector
from duplicate_removal.grouping import group_observations, manual_merge
from evaluation.metrics import count_metrics, detection_metrics
from maturity.assessment import assess, harvest_summary
from modules.inputs import field_records, projections_from_json, spatial_records, count_records
from modules.pipeline import process_images, process_video, annotate, finalize, refresh_artifacts
from modules.schema import Run
from preprocessing.images import PreprocessConfig, read_image
from reports.export import artifact_zip, write_reports
from training.model_info import trained_model_info


ROOT = Path(__file__).resolve().parent
TRAINED_INFO = trained_model_info(ROOT / "models")
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
logging.basicConfig(filename=OUTPUT / "app.log", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Mango Smart Counting", page_icon="🥭", layout="wide")
st.markdown("""<style>
.block-container {max-width:1440px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {border-right:1px solid #dce5dc}
[data-testid="stMetric"] {background:white;border:1px solid #dfe7dc;border-radius:14px;padding:16px}
[data-testid="stMetricValue"] {color:#205f3b}
.hero {padding:26px 30px;background:linear-gradient(115deg,#153e2c,#2c7147);border-radius:20px;color:white;margin-bottom:24px}
.hero h1 {color:white;font-size:2rem;margin:8px 0}
.hero p {color:#e0eddf;margin:4px 0;max-width:760px}
.eyebrow {font-size:.75rem;letter-spacing:.16em;color:#bed9ae}
.hint {font-size:.9rem;color:#56675c;line-height:1.65}
div.stButton > button {border-radius:9px;min-height:42px}
@media(max-width:640px){.hero {padding:20px}.hero h1 {font-size:1.5rem}.block-container {padding-top:1rem}}
</style>""", unsafe_allow_html=True)

PAGES = ["Dashboard", "Akuisisi & Analisis", "Asosiasi & ID", "Kematangan & Panen", "Evaluasi", "Riwayat & Ekspor", "Dataset & Model"]


def navigate(page):
    st.session_state["page"] = page


def activate(run):
    st.session_state["run_path"] = str(Path(run.directory) / "run.json")


def active_run():
    path = st.session_state.get("run_path")
    if not path:
        return None
    try:
        return Run.load(Path(path))
    except (OSError, ValueError, TypeError):
        st.error("Riwayat aktif tidak dapat dibaca. Pilih sesi lain di Riwayat & Ekspor.")
        return None


def report_error(exc):
    logger.exception("Application operation failed")
    st.error(f"Proses gagal: {exc}")


def progress_callback(bar):
    return lambda value, message: bar.progress(min(max(float(value), 0), 1), text=message)


def demo(video=False):
    bar = st.progress(0, text="Menyiapkan data uji sintetis")
    try:
        images = generate(ROOT / "dataset" / "synthetic")
        if video:
            path = ROOT / "dataset" / "synthetic" / "SYNTHETIC_orbit.mp4"
            run = process_video(path.read_bytes(), ".mp4", Detector(), OUTPUT,
                                PreprocessConfig(enhance=False), progress=progress_callback(bar))
        else:
            run = process_images([(p.name, p.read_bytes()) for p in images], Detector(), OUTPUT,
                                 PreprocessConfig(enhance=False), progress_callback(bar))
        activate(run)
        st.success("Demo sintetis selesai. Ini data uji buatan, bukan hasil kebun.")
        return run
    except Exception as exc:
        report_error(exc)
        return None


with st.sidebar:
    st.markdown("## 🥭 Mango Smart")
    st.caption("COUNTING SYSTEM · LAB PENGOLAHAN CITRA")
    page = st.radio("Navigasi", PAGES, key="page")
    st.divider()
    st.markdown("**Mesin deteksi**")
    mode_label = st.selectbox("Mode", ["Sintetis · data uji", "YOLO umum · demo COCO", "YOLO mangga · model sendiri"], index=2 if TRAINED_INFO else 0)
    mode = {"Sintetis · data uji": "synthetic", "YOLO umum · demo COCO": "yolo-demo", "YOLO mangga · model sendiri": "mango"}[mode_label]
    confidence = st.slider("Confidence / skor bentuk sintetis", 0.05, 0.95,
                           float(TRAINED_INFO["confidence_threshold"]) if TRAINED_INFO and mode == "mango" else 0.25, 0.01)
    model_path = ROOT / "models" / "best.pt"
    trusted_model = False
    class_ids = None
    if mode == "mango":
        local_model = st.text_input("Path model lokal", value=str(model_path))
        uploaded_model = st.file_uploader("Unggah model YOLO .pt", type=["pt"])
        class_text = st.text_input("ID kelas mangga (opsional)", placeholder="0 atau 0,1")
        model_path = Path(local_model)
        if TRAINED_INFO and model_path.resolve() == (ROOT / "models" / "best.pt").resolve() and uploaded_model is None:
            trusted_model = True
            st.success("Model mangga hasil training lokal aktif")
            st.caption("Dilakukan fine-tuning pada mangga di kebun, foto smartphone, mangga dipetik, dan contoh non-mangga. Hasil uji tersedia di Dataset & Model.")
        else:
            trust_key = hashlib.sha256((str(model_path)+(uploaded_model.name if uploaded_model else "")).encode()).hexdigest()[:12]
            trusted_model = st.checkbox("Saya mempercayai asal model ini", key="trust_"+trust_key)
            st.caption("Checkpoint PyTorch dapat memuat kode. Gunakan model dari sumber tepercaya.")
    st.divider()
    st.caption("Satu sesi = satu pohon atau satu kelompok buah setelah dipetik. Buah yang tidak terlihat belum teramati.")


def get_detector():
    if mode == "synthetic":
        raise ValueError("Mode sintetis hanya untuk tombol demo. Pilih mode YOLO untuk unggahan lapangan.")
    if mode == "mango":
        if not trusted_model:
            raise ValueError("Konfirmasikan bahwa sumber model tepercaya sebelum memuat checkpoint.")
        selected_path = model_path
        if uploaded_model is not None:
            data = uploaded_model.getvalue()
            if not data:
                raise ValueError("Model unggahan kosong.")
            selected_path = ROOT / "models" / f"user_{hashlib.sha256(data).hexdigest()[:16]}.pt"
            selected_path.write_bytes(data)
        ids = [int(value.strip()) for value in class_text.split(",") if value.strip()] if class_text else None
        return Detector("mango", selected_path, ids, confidence)
    return Detector("yolo-demo", ROOT / "models" / "yolo11n.pt", confidence=confidence)


def need_run(run):
    if run is None:
        st.info("Belum ada sesi aktif. Unggah mangga di pohon / setelah dipetik, atau coba data uji sintetis.")
        st.button("Buka akuisisi", on_click=navigate, args=("Akuisisi & Analisis",))
        return False
    return True


def run_notice(run):
    if run.mode == "synthetic":
        st.warning("DATA UJI SINTETIS — bentuk buatan, skor bentuk, dan proksi kematangan. Bukan deteksi mangga / hasil lapangan.")
    elif run.mode == "yolo-demo":
        st.warning("DEMO YOLO COCO — mendeteksi apple/banana/orange. Angka berikut bukan jumlah mangga.")
    else:
        st.info("Prediksi mangga yang terlihat. Model dapat dipakai untuk buah di pohon maupun setelah dipetik; akurasi pada sesi ini tetap memerlukan ground truth.")


def view_compare(run, key):
    selection = st.selectbox("Gambar / frame", [v.id for v in run.views], key=key)
    view = next(v for v in run.views if v.id == selection)
    obs = [o for o in run.observations if o.view_id == selection]
    before, after = st.columns(2)
    before.image(annotate(read_image(view.path), obs, unique=False), channels="BGR", caption=f"Sebelum asosiasi · {len(obs)} deteksi", width="stretch")
    after.image(str(Path(run.directory) / "annotated" / f"{view.id}.jpg"), caption="Sesudah asosiasi · ID buah konsisten", width="stretch")
    st.caption(f"{view.id} · {view.source} · {view.width} × {view.height} · Blur score {view.blur_score:.1f}")


def dashboard(run):
    st.markdown('''<div class="hero"><div class="eyebrow">VISION FOR BETTER HARVESTS</div>
    <h1>Kenali mangga, dari kebun hingga meja.</h1>
    <p>Kenali mangga di pohon maupun setelah dipetik, hubungkan pengamatan antarsudut, dan periksa hasil deteksi serta data panen.</p></div>''', unsafe_allow_html=True)
    if run is None:
        a, b = st.columns([1.4, 1])
        with a:
            st.subheader("Mangga di pohon atau setelah dipetik")
            st.write("Unggah satu foto untuk mendeteksi mangga, termasuk buah di tangan, meja, atau keranjang. Untuk foto berulang dari objek yang sama, tinjau asosiasi agar buah tidak dihitung dua kali.")
            st.button("Mulai deteksi mangga", type="primary", on_click=navigate, args=("Akuisisi & Analisis",))
            if st.button("Coba data uji sintetis"):
                if demo():
                    st.rerun()
        with b:
            preview = ROOT / "dataset" / "synthetic" / "SYNTHETIC_view_1.jpg"
            if preview.exists():
                st.image(str(preview), caption="Contoh sintetis: tiga bentuk bertekstur, bukan foto mangga.")
        return
    run_notice(run)
    summary = harvest_summary(run.fruits, run.settings.get("average_weight_g"))
    if run.mode == "mango":
        if run.observations:
            st.success(f"Mangga terdeteksi · {len(run.observations)} pengamatan dengan confidence ≥ {run.settings['confidence']:.2f}.")
        else:
            st.warning("Tidak ada mangga terdeteksi pada threshold ini. Ini belum membuktikan bahwa gambar pasti tidak mengandung mangga; periksa foto dan threshold.")
    metrics = st.columns(4)
    for col, label, value in zip(metrics, ["Gambar / frame", "Seluruh deteksi", "Estimasi unik terlihat", "Pengamatan duplikat"],
                                 [len(run.views), len(run.observations), len(run.fruits), len(run.observations)-len(run.fruits)]):
        col.metric(label, value)
    st.caption("Pengamatan duplikat = deteksi yang tergabung kembali ke buah yang sama; pada video termasuk kemunculan antarframe.")
    evaluated = [item for item in run.evaluations if item.get("kind") == "detection"]
    if evaluated:
        cols = st.columns(3)
        for col, name in zip(cols, ["Precision", "Recall", "F1"]):
            value = evaluated[-1].get(name)
            col.metric(f"{name} terhadap ground truth", f"{100*value:.1f}%" if value is not None else "—")
    else:
        st.caption("Akurasi deteksi belum dievaluasi: masukkan ground truth pada halaman Evaluasi.")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Distribusi kematangan")
        if run.fruits:
            df = pd.DataFrame(run.fruits)
            counts = df.groupby(["maturity", "maturity_basis"]).size().reset_index(name="jumlah")
            chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("maturity:N", title="Kelas"), y=alt.Y("jumlah:Q", title="Estimasi buah"),
                color=alt.Color("maturity_basis:N", title="Dasar klasifikasi"), tooltip=["maturity", "maturity_basis", "jumlah"])
            st.altair_chart(chart, width="stretch")
        else:
            st.info("Tidak ada objek terdeteksi. Periksa model, confidence, dan kualitas gambar.")
    with right:
        st.subheader("Kesiapan panen")
        st.metric("Siap panen terasesmen", summary["ready"])
        st.write(f"Belum siap: **{summary['not_ready']}** · Belum dinilai: **{summary['unassessed']}**")
        percent = summary["readiness_percent"]
        st.write(f"Siap / seluruh estimasi unik: **{percent:.1f}%**" if percent is not None else "Persentase belum tersedia.")
        st.caption("Buah tanpa data lapangan tidak dimasukkan sebagai belum siap. Klasifikasi visual perlu validasi varietas.")
        st.button("Lengkapi asesmen panen", on_click=navigate, args=("Kematangan & Panen",))
    st.subheader("Jejak pengamatan")
    view_compare(run, "dashboard_view")
    with st.expander(f"Catatan kualitas & batas analisis ({len(run.warnings)})"):
        for warning in run.warnings:
            st.write(f"• {warning}")
        st.write("Pohon bukan bidang datar. Perubahan sudut besar, buah mirip, angin, dan oklusi dapat memecah atau salah menggabungkan ID. Tinjau asosiasi dan hitungan manual.")


def acquisition():
    st.title("Akuisisi & Analisis")
    st.write("Unggah mangga di pohon atau setelah dipetik. Satu foto cukup untuk deteksi; foto bertumpang tindih dan gerakan perlahan membantu menghitung buah yang sama dari beberapa sudut.")
    if mode == "synthetic":
        st.info("Mode sintetis aktif. Jalankan fixture di bawah atau pilih model YOLO pada sidebar untuk data sendiri.")
        if st.button("Jalankan demo sintetis", type="primary"):
            demo()
        if st.button("Demo video sintetis · ByteTrack"):
            demo(video=True)
    if mode == "yolo-demo":
        st.warning("Model COCO tidak mempunyai kelas mangga. Demo ini hanya menguji jalur inferensi objek apple/banana/orange.")
    source = st.radio("Sumber", ["Foto multi sudut", "Video mengelilingi pohon", "Kamera browser"], horizontal=True)
    inputs, video = [], None
    if source == "Foto multi sudut":
        uploads = st.file_uploader("Foto JPG / PNG / WebP (maks. 40, urutkan berdasarkan sudut)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
        inputs = [(u.name, u.getvalue()) for u in uploads]
        for col, upload in zip(st.columns(3), uploads[:3]):
            try:
                col.image(upload.getvalue(), caption=upload.name, width="stretch")
            except Exception:
                col.warning("Preview gambar tidak tersedia.")
    elif source == "Video mengelilingi pohon":
        video = st.file_uploader("Video MP4 / MOV / AVI / MKV (maks. 200 MB)", type=["mp4", "mov", "avi", "mkv"])
        if video:
            st.video(video)
    else:
        st.caption("Kamera laptop memakai localhost. Ponsel memerlukan HTTPS tepercaya; IP LAN HTTP biasanya hanya mendukung unggah file, bukan izin kamera browser.")
        shot = st.camera_input("Ambil foto dengan kamera perangkat")
        if st.button("Tambahkan foto ke koleksi", disabled=shot is None):
            shots = st.session_state.setdefault("shots", [])
            if len(shots) >= 40:
                st.error("Maksimal 40 foto per koleksi.")
            else:
                shots.append((f"camera_{len(shots)+1}.jpg", shot.getvalue()))
        inputs = st.session_state.get("shots", [])
        st.write(f"{len(inputs)} foto tersimpan dalam koleksi sesi browser ini.")
        if st.button("Kosongkan koleksi kamera", disabled=not inputs):
            st.session_state["shots"] = []
            st.rerun()
        if inputs:
            st.image([data for _, data in inputs[-3:]], width=180)
        with st.expander("Kamera gagal terbuka / panduan ponsel"):
            st.write("Izinkan kamera pada browser dan OS. Buka aplikasi langsung, bukan iframe. Gunakan HTTPS dengan sertifikat yang dipercaya ponsel. Jika kamera tetap tidak tersedia, ambil foto memakai aplikasi kamera ponsel lalu unggah.")
    with st.expander("Preprocessing & batas komputasi", expanded=True):
        a, b, c = st.columns(3)
        max_side = a.select_slider("Sisi terpanjang (px)", [640, 960, 1280, 1600], value=960)
        enhance = b.checkbox("Peningkatan kontras CLAHE", value=mode != "mango")
        brightness = c.slider("Pengali pencahayaan", 0.3, 2.5, 1.0, 0.1)
        crop = None
        if st.checkbox("Crop proporsional"):
            crop_x = st.slider("Rentang horizontal (%)", 0, 100, (0, 100))
            crop_y = st.slider("Rentang vertikal (%)", 0, 100, (0, 100))
            crop = (crop_x[0]/100, crop_y[0]/100, crop_x[1]/100, crop_y[1]/100)
        stride = st.number_input("Proses setiap N frame (1 = setiap frame)", 1, 30, 1)
        cap = st.number_input("Batas frame video terproses", 1, 1000, 300, step=1)
        st.caption("Koordinat deteksi dan ground truth pada aplikasi menggunakan gambar setelah crop / resize. Stride dan batas frame dapat melewatkan buah.")
    if st.button("Deteksi & analisis mangga", type="primary", disabled=not(inputs or video)):
        bar = st.progress(0, text="Memuat model")
        try:
            detector = get_detector()
            config = PreprocessConfig(max_side, enhance, brightness, crop)
            if video:
                run = process_video(video.getvalue(), Path(video.name).suffix, detector, OUTPUT, config, stride, cap, progress_callback(bar))
            else:
                run = process_images(inputs, detector, OUTPUT, config, progress_callback(bar))
            activate(run)
            st.success(f"Analisis selesai: {len(run.views)} gambar/frame, {len(run.fruits)} estimasi objek unik terlihat.")
        except Exception as exc:
            report_error(exc)
    st.button("Lihat dashboard", on_click=navigate, args=("Dashboard",))


def associations(run):
    st.title("Asosiasi & ID buah")
    if not need_run(run):
        return
    run_notice(run)
    st.write("SIFT + Hungarian menghubungkan buah dengan bukti fitur lokal. ByteTrack dibantu gerak dan kemiripan tampilan; ID tracking saja tidak menentukan buah unik.")
    view_compare(run, "association_view")
    st.dataframe(pd.DataFrame([asdict(o) for o in run.observations]), hide_index=True, width="stretch")
    links = pd.DataFrame([asdict(link) for link in run.links])
    st.subheader("Tinjau bukti penggabungan")
    if not links.empty:
        signature = hashlib.sha256(links.to_json().encode()).hexdigest()[:12]
        edited = st.data_editor(links, disabled=["a", "b", "score", "method"], hide_index=True, key=f"links_{run.id}_{signature}")
        if st.button("Simpan keputusan asosiasi"):
            try:
                for link, accepted in zip(run.links, edited["accepted"]):
                    link.accepted = bool(accepted)
                group_observations(run.observations, run.links)
                run.fruits = assess(run.observations)
                run.settings.pop("calibration", None)
                run.settings.pop("field_records", None)
                run.evaluations = []
                refresh_artifacts(run)
                st.success("Asosiasi disimpan. ID dihitung ulang; data panen dan evaluasi sebelumnya perlu ditinjau ulang.")
                st.rerun()
            except Exception as exc:
                report_error(exc)
    else:
        st.info("Belum ada pasangan dengan bukti cukup. ID terpisah bukan bukti bahwa semua buah berbeda; tinjau foto dan gabungkan manual bila terkonfirmasi.")
    with st.expander("Gabungkan dua ID setelah pemeriksaan manual"):
        ids = [fruit["unique_id"] for fruit in run.fruits]
        if len(ids) >= 2:
            a, b = st.columns(2)
            first_id = a.selectbox("Buah pertama", ids)
            second_id = b.selectbox("Buah kedua", ids, index=1)
            if st.button("Gabungkan ID terkonfirmasi"):
                try:
                    manual_merge(run.observations, run.links, first_id, second_id)
                    run.fruits = assess(run.observations)
                    run.evaluations = []
                    run.settings.pop("calibration", None)
                    run.settings.pop("field_records", None)
                    refresh_artifacts(run)
                    st.rerun()
                except Exception as exc:
                    report_error(exc)
        else:
            st.caption("Minimal dua ID diperlukan.")
    with st.expander("Metadata kamera / posisi 3D opsional"):
        st.write("JSON: view_id → matriks proyeksi 3×4 dalam koordinat piksel gambar terproses. CSV 3D: observation_id,x,y,z dalam satu sistem koordinat. Data ini harus berasal dari kalibrasi / rekonstruksi nyata.")
        camera = st.file_uploader("Matriks kamera JSON", type="json")
        spatial = st.file_uploader("Posisi buah 3D CSV", type="csv")
        radius = st.number_input("Radius asosiasi 3D (satuan metadata)", min_value=0.0001, value=0.04, format="%.4f")
        if st.button("Hitung ulang dengan metadata", disabled=not(camera or spatial)):
            bar = st.progress(0)
            try:
                updated = copy.deepcopy(run)
                projections = projections_from_json(camera.getvalue()) if camera else None
                if spatial:
                    spatial_records(spatial.getvalue(), updated.observations)
                updated.settings.pop("calibration", None)
                updated.evaluations = []
                updated.settings["spatial_radius"] = radius
                if projections:
                    updated.settings["projections"] = {k: v.tolist() for k, v in projections.items()}
                finalize(updated, projections, radius, progress_callback(bar))
                st.success("Asosiasi dihitung ulang. Lengkapi ulang data panen karena ID dapat berubah.")
                st.rerun()
            except Exception as exc:
                report_error(exc)


def maturity_page(run):
    st.title("Kematangan & estimasi panen")
    if not need_run(run):
        return
    run_notice(run)
    st.info("Unripe / Mature / Ripe tanpa data lapangan adalah proksi tampilan yang belum tervalidasi. Harvest-ready adalah status terpisah, dapat beririsan dengan Mature / Ripe.")
    st.write("Isi CSV per unique_id. Kolom: unique_id,maturity,days_after_flowering,brix,firmness_n,weight_g. Kelas lapangan: Unripe, Mature, Ripe.")
    template = "unique_id,maturity,days_after_flowering,brix,firmness_n,weight_g\n" + "".join(f"{fruit['unique_id']},,,,,\n" for fruit in run.fruits)
    st.download_button("Unduh template data lapangan", template, "field_template.csv", "text/csv")
    fields_upload = st.file_uploader("Data pemeriksaan buah CSV", type="csv")
    with st.form("harvest_assessment"):
        a, b = st.columns(2)
        variety = a.text_input("Varietas yang diperiksa", placeholder="Nama varietas kebun Anda")
        min_days = b.number_input("Minimum hari setelah pembungaan", 0.0, 500.0, 0.0)
        min_brix = a.number_input("Minimum Brix", 0.0, 50.0, 0.0)
        max_firmness = b.number_input("Maksimum kekerasan (N)", 0.0, 500.0, 0.0)
        calibrated = st.checkbox("Ambang varietas di atas telah divalidasi di lapangan")
        avg_weight = st.number_input("Bobot rata-rata sampel (gram), 0 = belum diukur", 0.0, 5000.0, float(run.settings.get("average_weight_g") or 0))
        submitted = st.form_submit_button("Simpan asesmen panen", type="primary")
    if submitted:
        try:
            fields = field_records(fields_upload.getvalue()) if fields_upload else {}
            calibration = {"variety": variety, "min_days": min_days, "min_brix": min_brix, "max_firmness": max_firmness, "validated": True} if calibrated else None
            run.fruits = assess(run.observations, fields, calibration)
            run.settings["calibration"] = calibration
            run.settings["field_records"] = fields
            run.settings["average_weight_g"] = avg_weight or None
            run.save()
            write_reports(run)
            st.success("Asesmen disimpan. Tanpa ambang tervalidasi dan data lengkap, kesiapan tetap belum diketahui.")
        except Exception as exc:
            report_error(exc)
    summary = harvest_summary(run.fruits, run.settings.get("average_weight_g"))
    a, b, c, d = st.columns(4)
    a.metric("Siap panen", summary["ready"])
    b.metric("Belum siap", summary["not_ready"])
    c.metric("Belum dinilai", summary["unassessed"])
    d.metric("Estimasi bobot terlihat", f"{summary['estimated_total_kg']:.2f} kg" if summary["estimated_total_kg"] is not None else "—")
    st.caption(f"Bobot siap panen: {summary['estimated_ready_kg']:.2f} kg berdasarkan bobot rata-rata sampel." if summary["estimated_ready_kg"] is not None else "Masukkan bobot rata-rata sampel untuk estimasi kilogram. Tidak ada ekstrapolasi buah tersembunyi.")
    if run.fruits:
        table = pd.DataFrame(run.fruits)
        st.dataframe(table, hide_index=True, width="stretch")
        counts = Counter(fruit["maturity"] for fruit in run.fruits)
        st.dataframe(pd.DataFrame([{"kelas": cls, "jumlah": n, "persentase": n/len(run.fruits)*100} for cls, n in counts.items()]), hide_index=True)


def evaluation_page(run):
    st.title("Evaluasi dengan ground truth")
    if not need_run(run):
        return
    run_notice(run)
    st.write("Metrik hanya dihitung dari input aktual. Hitungan manual tidak dapat menghasilkan precision, recall, atau mAP tanpa anotasi bounding box.")
    actual = st.number_input("Hitungan manual buah unik terlihat pada sesi ini", 0, 100000, 0)
    if st.button("Evaluasi hitungan sesi"):
        result = {"kind": "count-session", "actual": actual, "predicted": len(run.fruits), **count_metrics([actual], [len(run.fruits)])}
        run.evaluations.append(result)
        run.save()
        write_reports(run)
        st.json(result)
    with st.expander("Evaluasi banyak pohon / sesi"):
        st.caption("CSV: tree_id,actual,predicted. Satu baris per pohon; MAPE mengabaikan actual=0 dan melaporkan jumlah baris tersebut.")
        counts = st.file_uploader("Ground truth hitungan CSV", type="csv")
        if st.button("Hitung MAE / RMSE / MAPE", disabled=counts is None):
            try:
                rows = count_records(counts.getvalue())
                result = {"kind": "count-batch", **count_metrics([r["actual"] for r in rows], [r["predicted"] for r in rows])}
                run.evaluations.append(result)
                run.save()
                write_reports(run)
                st.json(result)
            except Exception as exc:
                report_error(exc)
    with st.expander("Evaluasi bounding box · Precision / Recall / F1 / mAP@50", expanded=True):
        st.code('{"V0001": [[x1,y1,x2,y2]], "V0002": []}', language="json")
        st.caption("Sertakan setiap view, termasuk yang tidak mengandung buah. Koordinat piksel harus sesuai gambar terproses yang dapat diunduh pada paket ZIP.")
        truth = st.file_uploader("Ground truth bounding box JSON", type="json")
        if st.button("Evaluasi deteksi", disabled=truth is None):
            try:
                if len(truth.getvalue()) > 5_000_000:
                    raise ValueError("Anotasi melebihi 5 MB.")
                gt = json.loads(truth.getvalue())
                if not isinstance(gt, dict) or set(gt) != {v.id for v in run.views}:
                    raise ValueError("Ground truth harus mencakup tepat semua view sesi aktif.")
                predictions = [{"view_id": o.view_id, "box": o.box, "confidence": o.confidence} for o in run.observations]
                result = {"kind": "detection", **detection_metrics(predictions, gt)}
                run.evaluations.append(result)
                run.save()
                write_reports(run)
                st.json(result)
            except Exception as exc:
                report_error(exc)
    if run.evaluations:
        st.subheader("Riwayat evaluasi sesi")
        st.dataframe(pd.DataFrame(run.evaluations), hide_index=True, width="stretch")


def history_page(run):
    st.title("Riwayat & Ekspor")
    paths = sorted(OUTPUT.glob("*/run.json"), reverse=True)
    if paths:
        chosen = st.selectbox("Sesi tersimpan", paths, format_func=lambda p: p.parent.name)
        if st.button("Buka sesi terpilih"):
            try:
                selected = Run.load(chosen)
                activate(selected)
                st.rerun()
            except Exception as exc:
                report_error(exc)
    if not need_run(run):
        return
    run_notice(run)
    local_time = datetime.fromisoformat(run.created_at).astimezone(ZoneInfo("Asia/Jakarta"))
    st.write(f"**{run.id}** · {local_time:%d %b %Y %H:%M} WIB")
    st.caption(f"Disimpan lokal: {run.directory}")
    for col, filename, label, mime in zip(st.columns(4), ["fruits.csv", "observations.csv", "associations.csv", "report.pdf"],
                                          ["CSV buah unik", "CSV deteksi", "CSV asosiasi", "Laporan PDF"], ["text/csv"]*3+["application/pdf"]):
        file = Path(run.directory) / filename
        if file.exists():
            col.download_button(label, file.read_bytes(), filename, mime)
    if st.button("Siapkan paket lengkap ZIP"):
        with st.spinner("Mengemas gambar, video, metadata, dan laporan..."):
            st.download_button("Unduh paket lengkap", artifact_zip(run), f"{run.id}.zip", "application/zip")
    if run.video_path and Path(run.video_path).is_file():
        if Path(run.video_path).name.endswith("h264.mp4"):
            st.video(run.video_path)
        st.download_button("Unduh video tracking", Path(run.video_path).read_bytes(), "tracking.mp4", "video/mp4")
    selected_view = st.selectbox("Unduh hasil gambar", [v.id for v in run.views])
    path = Path(run.directory) / "annotated" / f"{selected_view}.jpg"
    st.image(str(path), caption=selected_view, width="stretch")
    st.download_button("Unduh gambar dengan ID unik", path.read_bytes(), path.name, "image/jpeg")


def dataset_page():
    st.title("Dataset & Model")
    status_path = OUTPUT / "training" / "queued_training.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["state"] == "waiting_resources":
                st.info("Training menunggu memori kosong: minimal RAM 4 GiB dan VRAM 4 GiB untuk GPU. Aplikasi lain tidak ditutup otomatis.")
            elif status["state"] == "training":
                st.info(f"Training nyata berlangsung · tahap {status['stage']}.")
                history = Path(status["run_path"]) / "results.csv"
                completed = len(pd.read_csv(history)) if history.exists() else 0
                st.progress(min(completed/status["epochs"], 1), text=f"{completed}/{status['epochs']} epoch tersimpan")
            elif status["state"] == "failed":
                st.error("Training belum berhasil. Checkpoint dan log disimpan; model final belum diperbarui.")
            elif status["state"] == "complete":
                st.success("Training dan evaluasi selesai; model hasil training tersedia.")
        except (OSError, ValueError, KeyError):
            st.caption("Status training sedang diperbarui.")
    if TRAINED_INFO:
        st.subheader("Hasil uji model terlatih")
        rows = []
        for domain in ["overall", "orchard", "smartphone-orchard", "postharvest"]:
            result = TRAINED_INFO[domain]
            metrics = result["standard_metrics"]
            precision = result["at_selected_threshold"]["Precision"]
            recall = result["at_selected_threshold"]["Recall"]
            rows.append({"domain": domain, "foto uji": result["images"], "mAP@50 (%)": metrics["metrics/mAP50(B)"]*100,
                         "mAP@50:95 (%)": metrics["metrics/mAP50-95(B)"]*100,
                         "Precision threshold (%)": precision*100 if precision is not None else None,
                         "Recall threshold (%)": recall*100 if recall is not None else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        negative = TRAINED_INFO["non_mango"]
        st.write(f"Non-mangga: {negative['images_with_false_detection']} dari {negative['images']} foto salah menghasilkan deteksi mangga pada threshold {TRAINED_INFO['confidence_threshold']:.2f}.")
        st.caption("Angka berasal dari test set publik terpisah. Bukan jaminan untuk semua varietas, latar, kamera, dan pencahayaan. Model ini tidak memprediksi kematangan fisiologis.")
        st.download_button("Unduh laporan evaluasi model JSON", json.dumps(TRAINED_INFO, indent=2), "model_evaluation.json", "application/json")
    st.write("Gunakan anotasi YOLO satu kelas: 0=mango. Pisahkan pohon / perekaman antar split agar frame serupa tidak bocor ke test.")
    st.code("dataset/raw/\n  images/tree01__angle1.jpg\n  labels/tree01__angle1.txt  # class x_center y_center width height (0..1)", language="text")
    source = st.text_input("Folder sumber dataset lokal", str(ROOT / "dataset" / "raw"))
    destination = st.text_input("Folder tujuan baru", str(ROOT / "dataset" / "prepared"))
    if st.button("Validasi & bagi train / val / test"):
        try:
            result = prepare(Path(source), Path(destination))
            st.success(f"Dataset siap: {result}. Sumber tetap tersimpan; tujuan berisi data.yaml.")
        except Exception as exc:
            report_error(exc)
    st.subheader("Training model mangga")
    st.code('yolo detect train model=yolo11n.pt data=dataset/prepared/data.yaml epochs=100 imgsz=640 device=cpu\nyolo detect val model=runs/detect/train/weights/best.pt data=dataset/prepared/data.yaml split=test conf=0.001', language="bash")
    st.caption("Training dijalankan lewat terminal karena dapat berlangsung lama. Gunakan device=0 bila PyTorch CUDA tersedia. Salin best.pt ke models/ lalu pilih mode model mangga.")
    if st.button("Siapkan ulang fixture sintetis"):
        try:
            generate(ROOT / "dataset" / "synthetic")
            st.success("Tiga foto, video, dan ground truth sintetis tersedia di dataset/synthetic.")
        except Exception as exc:
            report_error(exc)
    with st.expander("COLMAP & Open3D · integrasi opsional"):
        st.write("Integrasi belum aktif sampai COLMAP dan Open3D tersedia serta rekonstruksi berhasil. Jalur CLI berikut menjalankan COLMAP nyata, mengekspor PLY, dan membuka visualisasi Open3D.")
        st.code('python -m pip install -r requirements-3d.txt\npython -m multiview.reconstruction path/to/images output/reconstruction_new --show', language="bash")
        st.caption("Rekonstruksi pohon memerlukan tekstur dan overlap yang cukup. Titik COLMAP bukan otomatis posisi pusat buah. Untuk asosiasi 3D, masukkan posisi pengamatan buah yang sudah diregistrasi melalui CSV di Asosiasi & ID.")


run = active_run()
if page == "Dashboard":
    dashboard(run)
elif page == "Akuisisi & Analisis":
    acquisition()
elif page == "Asosiasi & ID":
    associations(run)
elif page == "Kematangan & Panen":
    maturity_page(run)
elif page == "Evaluasi":
    evaluation_page(run)
elif page == "Riwayat & Ekspor":
    history_page(run)
else:
    dataset_page()
