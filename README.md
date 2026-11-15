# Mango Smart Counting System

Aplikasi Python / Streamlit untuk mendeteksi objek, melacak video, mengasosiasikan pengamatan antarsudut, memperkirakan jumlah buah **unik yang terlihat**, dan menyimpan hasil yang dapat ditinjau. Antarmuka berbahasa Indonesia dengan tema pertanian hijau, grafik interaktif, kamera browser, riwayat lokal, dan ekspor.

**Status model:** tidak menyertakan model mangga terlatih atau klaim akurasi lapangan. Demo sintetis dapat dijalankan tanpa bobot YOLO. Model COCO resmi `yolo11n.pt` hanya mendeteksi apple/banana/orange untuk menguji jalur YOLO; objek tersebut tidak diubah label menjadi mangga. Gunakan `best.pt` hasil pelatihan mangga untuk foto pohon nyata.

Spesifikasi yang digunakan adalah teks permintaan proyek yang diberikan. Dokumen LMS asli “Menghitung Buah Mangga dari Satu Pohon Menggunakan Ponsel” tidak disertakan, sehingga kesesuaian dengan rubrik tambahan di LMS belum dapat diperiksa.

## Instalasi

Disarankan Python 3.11 atau 3.12, Windows 10/11 atau macOS, RAM 8 GB atau lebih. CPU didukung; pelatihan lebih cepat memakai GPU yang didukung PyTorch. Lingkungan pengembangan ini juga diuji dengan Python 3.14 dan PyTorch CPU.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Buka `http://localhost:8501`. Tidak memerlukan API key atau akun cloud. Instalasi awal dan unduhan bobot demo memerlukan internet; fixture sintetis bekerja offline setelah dependensi terpasang. Pada workspace yang telah disiapkan, `.venv` dan bobot demo tersedia.

Jika port dipakai, tambahkan `--server.port 8502`. Untuk kompatibilitas API ByteTrack, versi Ultralytics dikunci pada versi yang diuji. Jangan mencampur paket `opencv-python` dan `opencv-python-headless` pada environment yang sama. Lisensi Ultralytics adalah AGPL-3.0 / Enterprise; periksa ketentuannya bila aplikasi akan didistribusikan.

## Mulai cepat

1. Tekan **Coba data uji sintetis** pada Dashboard. Tiga foto berisi tiga bentuk bertekstur diproses menjadi sembilan deteksi dan tiga ID unik; ini pengujian buatan, bukan pengukuran kebun.
2. Periksa tampilan sebelum/sesudah, tabel deteksi, dan bukti penggabungan pada **Asosiasi & ID**.
3. Untuk foto sendiri, pilih **YOLO mangga · model sendiri**, masukkan model tepercaya, lalu unggah foto/video pada **Akuisisi & Analisis**.
4. Tinjau ID dan penggabungan manual sebelum memasukkan data lapangan pada **Kematangan & Panen**.
5. Masukkan hitungan manual / anotasi pada **Evaluasi**. Unduh hasil di **Riwayat & Ekspor**.

Untuk mencoba tracking tanpa model mangga, buka **Akuisisi & Analisis** dalam mode sintetis dan tekan **Demo video sintetis · ByteTrack**. Video uji berisi 24 frame dengan tiga bentuk yang sama, bukan pengambilan gambar kebun.

Satu sesi harus berasal dari satu pohon. Jangan mencampur pohon berbeda. Sistem tidak mengekstrapolasi jumlah mangga yang tidak pernah terlihat.

## Akuisisi, smartphone, dan preprocessing

- Foto: JPG/JPEG, PNG, WebP; maksimum 40 foto, 40 MB / foto, 40 megapiksel / foto, dan 200 MB total. Foto rusak ditolak dengan catatan; foto valid lainnya dapat diproses. Rotasi EXIF kamera ponsel diterapkan.
- Video: MP4, MOV, AVI, MKV; maksimum 200 MB. Default memproses setiap frame sampai 300 frame; batas dan stride dapat diubah. Batas, stride, dan video terpotong dinyatakan dalam hasil. Video yang gagal didekode ditolak.
- Kamera: `st.camera_input` memakai kamera perangkat yang **membuka browser**, termasuk laptop dan ponsel. Ambil beberapa foto dan tambahkan ke koleksi sebelum diproses. Ini akuisisi foto, bukan streaming webcam tanpa batas.
- Resize menjaga rasio, tidak memperbesar gambar kecil; CLAHE opsional, pengali pencahayaan, crop proporsional, dan skor blur Laplacian. Ambang blur 50 hanya peringatan awal dan perlu kalibrasi kamera. Fungsi normalisasi RGB CHW 0..1 tersedia; predictor YOLO menormalisasi input sendiri.
- Anotasi evaluasi dan metadata kamera harus menggunakan koordinat **gambar terproses setelah crop/resize**, bukan koordinat foto asli.

Untuk akses ponsel di Wi-Fi yang sama:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0
```

Lihat IP laptop dengan `ipconfig` (Windows) atau pengaturan jaringan (macOS), lalu buka `http://IP-LAPTOP:8501` di ponsel. Izinkan akses jaringan privat pada firewall bila diperlukan. **Unggah foto/video dapat bekerja melalui IP LAN HTTP, tetapi kamera browser umumnya membutuhkan secure context HTTPS.** `localhost` pada ponsel menunjuk ponsel itu sendiri.

Untuk kamera ponsel, gunakan reverse proxy HTTPS dengan sertifikat yang dipercaya perangkat, lalu buka URL HTTPS tersebut. Streamlit juga menerima `--server.sslCertFile path/to/cert.pem --server.sslKeyFile path/to/key.pem`, tetapi sertifikat harus dipercaya ponsel. Periksa izin kamera browser/OS dan hindari iframe yang tidak memberikan izin kamera. Bila kamera tidak tersedia, gunakan aplikasi kamera bawaan lalu unggah. Aplikasi ini tidak menyediakan autentikasi; batasi akses pada perangkat/jaringan tepercaya.

## Model YOLO dan training

Model lokal default: `models/best.pt`. Model dapat diunggah melalui sidebar atau dipilih lewat path lokal. Checkpoint PyTorch adalah input tepercaya karena dapat memuat kode; aplikasi meminta konfirmasi kepercayaan sebelum memuat model pengguna. Unggahan disimpan dengan nama berdasarkan hash, bukan nama file dari browser.

Kelas `mango` / `mangga` dikenali otomatis. Untuk nama kelas berbeda, tentukan ID kelas eksplisit, misalnya `0`. Model deteksi diperlukan; model klasifikasi saja ditolak. Confidence dapat diatur di sidebar. Metadata sesi mencatat kelas, nama model, dan SHA-256 checkpoint untuk membedakan hasil model.

Siapkan dataset beranotasi YOLO satu kelas:

```text
dataset/raw/
  images/
    tree01__angle1.jpg
    tree01__angle2.jpg
    tree02__angle1.jpg
    tree03__angle1.jpg
  labels/
    tree01__angle1.txt
    tree01__angle2.txt
    tree02__angle1.txt
    tree03__angle1.txt
```

Setiap baris label: `0 x_center y_center width height`, dinormalisasi 0..1. Gambar negatif harus memiliki file label kosong. Gunakan satu awalan pohon / perekaman sebelum `__` atau `_frame` untuk semua foto berkorelasi. Minimal tiga grup independen diperlukan. Pemisahan sekitar 70/20/10 dilakukan berdasarkan **grup**, sehingga proporsi gambar bisa berbeda. Gambar identik dalam grup berbeda ditolak untuk mencegah kebocoran; frame mirip yang bukan identik tetap perlu ditinjau manusia. Pembagian tidak menimpa folder tujuan yang sudah ada.

Jalankan dari halaman **Dataset & Model** atau terminal:

```bash
python -m dataset.prepare dataset/raw dataset/prepared
yolo detect train model=yolo11n.pt data=dataset/prepared/data.yaml epochs=100 imgsz=640 device=cpu
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/prepared/data.yaml split=test conf=0.001
```

Aktifkan environment dahulu atau gunakan `.venv/Scripts/yolo.exe` di Windows / `.venv/bin/yolo` di macOS. Gunakan `device=0` jika PyTorch CUDA sudah tersedia. Training dilakukan melalui terminal; aplikasi tidak menampilkan tombol training palsu. Evaluasi pada pohon, kamera, pencahayaan, dan varietas yang tidak bocor dari train. Salin bobot terbaik ke `models/best.pt` setelah training.

## Tracking dan penghapusan duplikasi

1. Deteksi disimpan sebagai pengamatan `V0001-D001`, dengan bbox, skor, kelas, dan ID tracking jika ada.
2. ByteTrack menggunakan Kalman filter dan asosiasi internal untuk menghubungkan deteksi video. Instance baru dibuat untuk setiap video. ID tracking hanya petunjuk, bukan hitungan aktual.
3. SIFT mengekstrak fitur, Lowe ratio test memilih pasangan, dan RANSAC memeriksa homografi jika bukti cukup. Pasangan buah memerlukan bukti fitur lokal, kemiripan tampilan, dan geometri. Track yang terputus masih dapat digabung melalui SIFT.
4. Hungarian melakukan pencocokan satu-ke-satu dan menolak pasangan yang ambigu. Kontinuitas ByteTrack hanya diterima dengan gerak / geometri, perubahan ukuran, dan tampilan yang masuk akal.
5. Komponen asosiasi membentuk ID buah `M0001`. Dua deteksi yang muncul bersamaan dalam satu view tidak boleh menjadi satu buah, termasuk melalui penggabungan transitif.
6. Pengguna dapat menolak link otomatis atau menggabungkan ID setelah pemeriksaan manual. Untuk memisahkan komponen, nonaktifkan semua jalur link yang menghubungkannya.

Semua pasangan foto (maksimum 40) dibandingkan. Video membandingkan tiga frame terproses sebelumnya dan frame pertama untuk loop closure. Cache fitur video dibatasi pada frame-frame tersebut; citra frame disimpan ke disk. Oklusi panjang dan kembalinya kamera ke sudut selain frame pertama dapat menghasilkan ID tambahan. Fitur ini adalah **estimasi asosiasi 2D konservatif**, bukan penyelesaian identitas global yang terjamin pada seluruh pohon 3D. Buah dengan tekstur kurang, sudut berubah besar, parallax, angin, atau warna serupa memerlukan review.

`seluruh deteksi - estimasi ID unik` adalah jumlah **pengamatan berulang yang berhasil digabung**. Pada video, angka ini termasuk buah yang muncul dalam banyak frame. Jika tidak ada bukti pencocokan, ID tetap terpisah; angka tersebut belum membuktikan bahwa semua buah berbeda. Koreksi asosiasi menghapus asesmen panen dan evaluasi lama karena ID dapat berubah.

Metadata opsional pada **Asosiasi & ID**:

- JSON kamera: `{"V0001": [[...], [...], [...]], "V0002": [[...], [...], [...]]}`. Setiap matriks proyeksi berukuran 3×4, rank 3, bernilai hingga. Kamera yang terkalibrasi menambahkan batas epipolar pada pasangan fitur.
- CSV posisi buah: `observation_id,x,y,z`. Koordinat harus berasal dari pengukuran atau rekonstruksi nyata, berada pada satu sistem koordinat, dan memakai satuan yang konsisten. Posisi dekat dapat diasosiasikan dalam radius yang dipilih; kendala satu buah per view tetap berlaku. Ini bukan estimasi koordinat 3D dari warna atau bbox.

## Kematangan dan panen

Tiga kelas kematangan dan status panen disimpan terpisah agar tidak menghitung buah yang sama dua kali:

| Nilai | Makna / dasar |
|---|---|
| Unripe | Belum matang; tanpa data lapangan hanya proksi visual |
| Mature | Matang fisiologis jika dikonfirmasi di lapangan; warna saja tidak membuktikannya |
| Ripe | Matang lanjut jika dikonfirmasi di lapangan |
| Harvest-ready | Status panen terpisah; memerlukan pemeriksaan dan kalibrasi varietas |

Proksi visual menghitung fraksi kuning pada area berwarna bbox dan menandai dasar `visual-proxy-unvalidated`. Ambang proksi tidak memiliki validasi agronomis. Varietas yang tetap hijau saat matang, pencahayaan, dan latar di bbox dapat menyesatkan. Jangan menggunakan proksi visual sebagai dasar keputusan panen.

Unduh template CSV sesuai ID sesi. Isi:

```csv
unique_id,maturity,days_after_flowering,brix,firmness_n,weight_g
M0001,Mature,110,14,20,400
```

Baris di atas hanya **contoh format**, bukan rekomendasi ambang varietas. Masukkan nama varietas, minimum hari setelah pembungaan, minimum Brix, dan maksimum kekerasan dari prosedur lapangan sendiri. Konfirmasikan ambang sudah divalidasi. Status siap hanya dihitung bila ada kelas kematangan lapangan, ketiga pengukuran, dan kalibrasi lengkap. Siap = kelas Mature/Ripe dan seluruh ambang terpenuhi. Pengukuran tidak lengkap menghasilkan **belum dinilai**, bukan belum siap.

Persentase kesiapan = siap / seluruh estimasi buah unik terlihat × 100; jumlah yang belum dinilai selalu ditampilkan. Kilogram dihitung menggunakan bobot rata-rata sampel yang dimasukkan pengguna, bukan bobot yang dikarang dari gambar. Bobot CSV per buah disimpan untuk pemeriksaan; estimasi pohon memakai rata-rata sampel yang dinyatakan eksplisit. Distribusi kematangan menghitung setiap ID sekali, menggunakan median proksi antarpengamatan atau kelas hasil pemeriksaan lapangan.

## Evaluasi

- Hitungan satu sesi: masukkan jumlah manual **buah unik terlihat pada cakupan yang sama**. Untuk jumlah aktual seluruh pohon termasuk buah tersembunyi, perbedaan mencakup keterbatasan akuisisi.
- Banyak pohon: CSV `tree_id,actual,predicted`, satu baris per pohon.
- MAE = rata-rata `abs(predicted-actual)`; RMSE = akar rata-rata kuadrat error.
- MAPE = rata-rata `abs(error)/actual × 100` untuk actual > 0. Baris actual=0 dikecualikan dan jumlahnya dilaporkan. Jika semua actual=0, MAPE tidak tersedia.
- Counting Error = jumlah signed error; Absolute Counting Error juga dilaporkan agar error positif/negatif tidak tersembunyi oleh pembatalan.
- Precision, Recall, F1, mAP@50 membutuhkan ground truth bbox, bukan hanya hitungan manual. Upload JSON `view_id -> daftar bbox xyxy` yang mencakup tepat seluruh view sesi, termasuk daftar kosong untuk gambar negatif. Gunakan gambar terproses di ZIP sebagai acuan.
- Prediksi diurutkan menurut confidence, dicocokkan satu-ke-satu ke GT pada IoU ≥ 0.5; prediksi ganda dihitung sebagai false positive. AP50 memakai interpolasi 101 titik, satu kelas. `mAP@50` di aplikasi berarti AP50 satu kelas pada prediksi yang tersimpan di threshold pilihan. Untuk penilaian model standar pada seluruh kurva confidence, gunakan `yolo detect val ... conf=0.001` pada test set.
- Jika tidak ada ground truth, aplikasi tidak menampilkan angka akurasi. Precision/Recall/F1 evaluasi yang tersedia ditampilkan dalam persen pada dashboard; tidak dilabeli sebagai akurasi keseluruhan pohon.

## Ekspor, riwayat, dan video

Setiap sesi tersimpan di `output/<run_id>/`:

```text
run.json                metadata, deteksi, link, ID, asesmen, evaluasi
views/V0001.jpg          gambar terproses sebagai acuan anotasi
annotated/V0001.jpg      bbox, skor, track ID, unique fruit ID
observations.csv        semua pengamatan
fruits.csv              satu baris per estimasi buah unik
associations.csv        bukti dan keputusan penggabungan
report.pdf              ringkasan, grafik, peringatan, evaluasi, contoh hasil
tracking.mp4            video tracking bila sumber video
tracking_h264.mp4       versi browser jika FFmpeg tersedia
```

Riwayat tetap tersedia setelah aplikasi direstart. Gambar asli tidak disimpan, tetapi gambar setelah preprocessing disimpan. Video input sementara dihapus setelah dekode, termasuk saat terjadi error. Paket ZIP berisi hasil, metadata, dan seluruh gambar terproses; perhatikan isinya sebelum dibagikan. CSV melindungi string yang dapat diperlakukan sebagai formula spreadsheet. Log lokal: `output/app.log`. JSON ditulis melalui file sementara lalu diganti agar riwayat tidak setengah tertulis.

FFmpeg bersifat opsional. OpenCV menulis `mp4v`; video dapat diunduh dan dibuka dengan VLC / pemutar yang mendukungnya. Bila executable `ffmpeg` tersedia di PATH, hasil juga dikonversi ke H.264 untuk preview browser. Tanpa FFmpeg, aplikasi menyatakan keterbatasan preview. Output video memuat frame terproses pada FPS sumber dibagi stride; tidak menyertakan audio.

## COLMAP dan Open3D (opsional)

Tidak aktif secara otomatis. Instal COLMAP terpisah dan pastikan `colmap` tersedia di PATH. Instal Open3D jika tersedia wheel untuk versi Python / OS Anda; Python 3.11/3.12 disarankan untuk fitur ini.

```bash
python -m pip install -r requirements-3d.txt
python -m multiview.reconstruction path/to/images output/reconstruction_new --show
```

Adapter menjalankan `feature_extractor`, `exhaustive_matcher`, `mapper`, dan `model_converter` nyata. Kegagalan executable atau rekonstruksi dilaporkan; output perintah tersimpan di `colmap.log`. Folder tujuan harus baru. Model pertama yang berhasil dikonversi ke text dan diekspor sebagai PLY melalui Open3D; `--show` membuka visualisasi desktop. Ini bukan rekonstruksi di browser dan belum divalidasi pada pohon nyata. GPU / driver COLMAP mungkin memerlukan penyesuaian lokal.

Titik sparse COLMAP bukan otomatis pusat buah, dan skala rekonstruksi monokular biasanya relatif. Registrasikan pusat buah dan satuan sebelum memakai CSV 3D. Tidak ada penghitungan buah dari banyaknya titik point cloud.

## Struktur

```text
app.py                  UI Streamlit dan navigasi
modules/                kontrak data, validasi CSV/JSON, pipeline, penyimpanan
detection/              inferensi YOLO dan mode sintetis
tracking/               ByteTrack nyata
preprocessing/          decode, EXIF, crop, resize, CLAHE, blur, normalisasi
multiview/              SIFT, Hungarian, epipolar, adapter COLMAP/Open3D
duplicate_removal/      constrained components, koreksi manual
maturity/               proksi visual, data lapangan, estimasi panen
evaluation/             bbox metrics dan count metrics
reports/                CSV, PDF, ZIP
models/                 best.pt pengguna dan bobot demo
dataset/                generator fixture dan pemisahan dataset
output/                 riwayat dan hasil runtime (diabaikan Git)
tests/                  unit, integrasi, dan Streamlit AppTest
```

## Pengujian

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tes menggunakan fixture sintetis dengan ground truth eksplisit. Mencakup preprocessing, EXIF, blur, input rusak, SIFT, Hungarian, asosiasi 3D / epipolar, konflik penggabungan transitif, tracklet terputus, ByteTrack nyata, reset tracker, hitungan unik foto/video, asesmen panen, metrics, dataset split, parser COLMAP, CSV/PDF/ZIP, persistensi, serta navigasi dan demo Streamlit.

Tes inferensi YOLO nyata memakai `models/yolo11n.pt`; bila bobot belum ada, tes tersebut ditandai **skip**, bukan dinyatakan berhasil. Unduh bobot demo tepercaya lalu jalankan ulang:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
.\.venv\Scripts\python.exe -m pytest -q
```

Tes ini memverifikasi jalur inferensi, bukan akurasi mangga. Kamera fisik, TLS ponsel, codec semua perangkat, macOS, GPU, training mangga, dan rekonstruksi nyata memerlukan pengujian perangkat / data terkait.

## Referensi implementasi

- [Ultralytics object tracking](https://docs.ultralytics.com/modes/track/)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/)
- [Streamlit camera_input](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input)
- [OpenCV feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [COLMAP command-line interface](https://colmap.github.io/cli.html)
- [Open3D visualization](https://www.open3d.org/docs/release/tutorial/visualization/visualization.html)
