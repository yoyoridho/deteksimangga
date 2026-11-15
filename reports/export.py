from collections import Counter
from dataclasses import asdict
from io import BytesIO, StringIO
from pathlib import Path
import csv
import zipfile
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

from maturity.assessment import harvest_summary


def csv_bytes(rows, columns):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        cleaned = {key: "'"+value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value for key, value in row.items()}
        writer.writerow(cleaned)
    return stream.getvalue().encode("utf-8-sig")


def pdf_bytes(run):
    stream = BytesIO()
    styles = getSampleStyleSheet()
    text = lambda value: Paragraph(escape(str(value)), styles["BodyText"])
    story = [Paragraph("Mango Smart Counting System", styles["Title"]), Spacer(1, 12),
             text(f"Run: {run.id} | UTC: {run.created_at} | Mode: {run.mode}"),
             text("Estimasi buah terlihat, bukan jaminan jumlah aktual pada seluruh pohon. Demo bukan hasil lapangan."), Spacer(1, 12)]
    summary = harvest_summary(run.fruits, run.settings.get("average_weight_g"))
    rows = [["Metrik", "Hasil"], ["Gambar / frame", str(len(run.views))], ["Seluruh deteksi", str(len(run.observations))],
            ["Estimasi buah unik terlihat", str(summary["unique"])], ["Pengamatan duplikat", str(len(run.observations)-summary["unique"])],
            ["Siap panen terasesmen", str(summary["ready"])], ["Belum siap terasesmen", str(summary["not_ready"])],
            ["Kesiapan belum diketahui", str(summary["unassessed"])],
            ["Siap / seluruh estimasi unik (%)", str(summary["readiness_percent"])],
            ["Estimasi bobot (kg)", str(summary["estimated_total_kg"])]]
    table = Table(rows, colWidths=[290, 170])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeadb")),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 12), text("Kematangan visual adalah proksi warna yang belum tervalidasi. Kesiapan panen memerlukan kelas lapangan, umur bunga, Brix, kekerasan, dan ambang varietas tervalidasi.")])
    if run.fruits:
        distribution = Counter(fruit["maturity"] for fruit in run.fruits)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(distribution.keys(), distribution.values(), color=["#4c956c", "#d2ae49", "#d97b40"][:len(distribution)])
        ax.set_ylabel("Estimasi buah terlihat")
        fig.tight_layout()
        chart = BytesIO()
        fig.savefig(chart, format="png", dpi=120)
        plt.close(fig)
        chart.seek(0)
        story.append(Image(chart, width=420, height=182))
    for warning in run.warnings:
        story.append(text(warning))
    for evaluation in run.evaluations:
        story.extend([Spacer(1, 10), text(f"Evaluasi: {evaluation}")])
    for view in run.views[:2]:
        path = Path(run.directory) / "annotated" / f"{view.id}.jpg"
        if path.exists():
            scale = min(420/view.width, 250/view.height)
            story.extend([Spacer(1, 10), text(view.id), Image(str(path), width=view.width*scale, height=view.height*scale)])
    SimpleDocTemplate(stream, title="Mango Smart Counting System", author="Mango Smart Counting System").build(story)
    return stream.getvalue()


def write_reports(run):
    root = Path(run.directory)
    (root / "observations.csv").write_bytes(csv_bytes([asdict(o) for o in run.observations],
        ["id", "view_id", "box", "confidence", "label", "track_id", "unique_id", "yellow_fraction", "xyz"]))
    (root / "fruits.csv").write_bytes(csv_bytes(run.fruits, ["unique_id", "observations", "maturity", "maturity_basis", "yellow_fraction",
        "harvest_ready", "variety", "days_after_flowering", "brix", "firmness_n", "weight_g"]))
    (root / "associations.csv").write_bytes(csv_bytes([asdict(link) for link in run.links], ["a", "b", "score", "method", "accepted"]))
    (root / "report.pdf").write_bytes(pdf_bytes(run))


def artifact_zip(run):
    stream = BytesIO()
    root = Path(run.directory)
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".pdf", ".jpg", ".mp4", ".ply"}:
                archive.write(path, path.relative_to(root).as_posix())
    return stream.getvalue()
