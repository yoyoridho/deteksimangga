# Mango Smart Counting System

Aplikasi Python / Streamlit untuk mendeteksi objek, melacak video, mengasosiasikan pengamatan antarsudut, memperkirakan jumlah buah **unik yang terlihat**, dan menyimpan hasil yang dapat ditinjau. Antarmuka berbahasa Indonesia dengan tema pertanian hijau, grafik interaktif, kamera browser, riwayat lokal, dan ekspor.

**Model mangga:** bobot hasil training disimpan di `models/best.pt`, disertai metrik dan SHA-256 pada `models/training_summary.json`. Jika keduanya tersedia dan cocok, aplikasi otomatis memilih model mangga dan threshold hasil validasi. Model mendeteksi mangga di pohon maupun setelah dipetik. Panduan sumber data, pemisahan, training, evaluasi, dan keterbatasannya ada di [training/README.md](training/README.md). Angka pada dataset uji bukan jaminan akurasi pada semua foto lapangan.

Demo sintetis dapat dijalankan tanpa bobot YOLO. Model COCO resmi `yolo11n.pt` hanya mendeteksi apple/banana/orange untuk menguji jalur YOLO; objek tersebut tidak diubah label menjadi mangga. Bila model mangga belum tersedia setelah clone, jalankan workflow training nyata atau masukkan model tepercaya sendiri. Dataset besar dan checkpoint tidak ikut diunggah melalui Git secara otomatis.

Spesifikasi yang digunakan adalah teks permintaan proyek yang diberikan. Dokumen LMS asli “Menghitung Buah Mangga dari Satu Pohon Menggunakan Ponsel” tidak disertakan, sehingga kesesuaian dengan rubrik tambahan di LMS belum dapat diperiksa.

## Instalasi lengkap: dari nol sampai aplikasi terbuka

Panduan ini ditujukan untuk **clone baru dari GitHub**. Ikuti urutannya; setiap langkah menyertakan perintah Windows dan macOS/Linux. Perkiraan waktu 10–30 menit, tergantung kecepatan internet. Tidak diperlukan API key, akun cloud, atau database.

### 1. Siapkan kebutuhan dasar

| Kebutuhan | Keterangan |
|---|---|
| Sistem operasi | Windows 10/11, macOS 12+, atau Linux |
| Python | **3.11 atau 3.12 disarankan.** Lingkungan pengembangan ini juga diuji dengan Python 3.14 + PyTorch CPU |
| Git | Untuk mengunduh repository. Alternatif: unduh ZIP lalu ekstrak |
| RAM | Minimal 8 GB |
| Ruang disk | Sekitar 2–3 GB untuk environment dan paket PyTorch |
| Internet | Diperlukan saat instalasi dependensi dan unduhan model demo |
| Browser | Chrome, Edge, Firefox, atau Safari versi terbaru |

Periksa Python dan Git sudah terpasang:

```powershell
python --version
git --version
```

Jika `python` tidak dikenali, instal Python dari [python.org](https://www.python.org/downloads/) dan centang **Add python.exe to PATH** saat instalasi. Di Windows, `py -3.12 --version` juga dapat dipakai bila launcher Python tersedia. Jika Git belum ada, instal dari [git-scm.com](https://git-scm.com/downloads).

### 2. Clone repository

```powershell
git clone https://github.com/yoyoridho/deteksimangga.git
cd deteksimangga
```

Jika memakai ZIP, ekstrak file, buka terminal di folder hasil ekstrak, lalu lanjut ke langkah berikut. Semua perintah di bawah harus dijalankan dari **root proyek** (folder yang berisi `app.py`).

### 3. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

macOS / Linux:

```bash
python3 -m venv .venv
```

Virtual environment memisahkan paket proyek dari Python sistem. Folder `.venv` tidak ikut masuk Git.

### 4. Instal dependensi

Pilih **salah satu** cara berikut.

**Cara A — tanpa aktivasi (disarankan, paling sedikit masalah di Windows):**

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

**Cara B — aktifkan environment lebih dulu:**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak `Activate.ps1` dengan pesan *running scripts is disabled*, izinkan hanya untuk sesi terminal saat ini:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`requirements.txt` memasang Streamlit, OpenCV, Ultralytics, PyTorch (wheel CPU), NumPy, pandas, SciPy, matplotlib, ReportLab, `lap`, dan pytest. Unduhan paket cukup besar; pastikan koneksi stabil. Jangan mencampur `opencv-python` dengan `opencv-python-headless` pada environment yang sama.

### 5. Verifikasi instalasi

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import streamlit, cv2, torch, ultralytics, pandas, reportlab; print('Streamlit', streamlit.__version__); print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

macOS / Linux:

```bash
.venv/bin/python -c "import streamlit, cv2, torch, ultralytics, pandas, reportlab; print('Streamlit', streamlit.__version__); print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

Tidak ada error berarti dependensi siap. `CUDA False` normal untuk instalasi CPU. Untuk memastikan seluruh kode berjalan, jalankan tes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tes sintetis tidak memerlukan model mangga. Tes inferensi YOLO nyata akan **skip** bila `models/yolo11n.pt` belum ada.

### 6. Jalankan aplikasi

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
.venv/bin/python -m streamlit run app.py
```

Tunggu hingga terminal menampilkan:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Buka **http://localhost:8501** di browser. Biasanya browser terbuka otomatis; jika tidak, salin URL tersebut secara manual. Biarkan terminal tetap berjalan selama aplikasi dipakai. Hentikan dengan `Ctrl+C`.

Jika port 8501 dipakai proses lain:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

lalu buka `http://localhost:8502`. Jangan membuka `0.0.0.0` di browser; alamat itu hanya untuk mode akses jaringan.

### 7. Pastikan program berhasil terbuka

Instalasi dinyatakan berhasil bila seluruh pemeriksaan berikut terpenuhi:

1. Halaman **Dashboard** tampil dengan judul “Kenali mangga, dari kebun hingga meja.” dan sidebar berisi **Navigasi** serta **Mesin deteksi**.
2. Tidak ada traceback atau pesan `ModuleNotFoundError` di halaman maupun terminal.
3. Tombol **Coba data uji sintetis** dapat diklik. Hasilnya menampilkan 3 gambar, 9 deteksi, dan 3 estimasi buah unik.
4. Menu **Akuisisi & Analisis**, **Asosiasi & ID**, **Kematangan & Panen**, **Evaluasi**, **Riwayat & Ekspor**, dan **Dataset & Model** dapat dibuka tanpa error.
5. Pada **Akuisisi & Analisis** mode sintetis, tombol **Demo video sintetis · ByteTrack** berjalan dan menghasilkan video tracking.

Demo sintetis bekerja tanpa bobot YOLO dan tanpa internet setelah dependensi terpasang. Ini adalah uji jalur aplikasi, bukan pengukuran kebun.

### 8. Tambahkan model (opsional, untuk foto mangga nyata)

Repository tidak menyertakan checkpoint `.pt` karena ukuran, lisensi, dan alasan reproduksibilitas. Tanpa model, aplikasi tetap terbuka dalam mode sintetis.

Untuk memakai model mangga hasil training sendiri:

1. Latih model mengikuti [training/README.md](training/README.md), atau letakkan model tepercaya di `models/best.pt`.
2. Jika tersedia `models/training_summary.json` dengan SHA-256 yang cocok, aplikasi otomatis memilih **YOLO mangga · model sendiri** dan threshold hasil validasi.
3. Alternatif: unggah file `.pt` melalui sidebar, centang **Saya mempercayai asal model ini**, lalu proses foto di **Akuisisi & Analisis**.

Untuk sekadar menguji jalur YOLO dengan model COCO resmi (hanya mendeteksi apple/banana/orange, bukan mangga):

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt')"
```

Checkpoint PyTorch dapat memuat kode; hanya muat model dari sumber yang Anda percayai.

### 9. Akses dari ponsel di Wi-Fi yang sama (opsional)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0
```

Cari IP laptop dengan `ipconfig` (Windows) atau pengaturan jaringan (macOS), lalu buka `http://IP-LAPTOP:8501` dari ponsel. Izinkan akses jaringan privat pada firewall bila diminta. Unggah foto/video dapat berjalan lewat HTTP LAN, tetapi **kamera browser umumnya memerlukan HTTPS**; bila kamera tidak tersedia, gunakan aplikasi kamera bawaan lalu unggah fotonya. Aplikasi tidak menyediakan autentikasi, jadi batasi akses pada jaringan tepercaya.

### 10. GPU untuk training atau inferensi cepat (opsional)

Aplikasi berjalan dengan CPU. Untuk GPU NVIDIA, buat environment terpisah agar wheel CUDA tidak bercampur dengan environment aplikasi:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

`torch.cuda.is_available()` harus `True` bila driver NVIDIA dan GPU didukung. Gunakan interpreter `.venv-gpu` untuk training, dan `.venv` untuk menjalankan aplikasi. Detail batasan GPU Pascal ada di [training/README.md](training/README.md).

### 11. Pemecahan masalah instalasi

| Gejala | Penyebab umum / solusi |
|---|---|
| `python` tidak dikenali | Python belum masuk PATH. Instal ulang dengan opsi PATH, atau pakai `py -3.12`. |
| `Activate.ps1 cannot be loaded` | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu aktifkan lagi. Atau gunakan Cara A tanpa aktivasi. |
| `No module named streamlit` / `ModuleNotFoundError` | Perintah dijalankan dengan Python sistem, bukan `.venv`. Selalu gunakan `.\.venv\Scripts\python.exe` (Windows) atau `.venv/bin/python` (macOS/Linux). |
| Instalasi paket berhenti / timeout | Periksa koneksi, lalu ulangi `pip install -r requirements.txt`. Untuk jaringan lambat tambahkan `--timeout 120`. |
| PyTorch gagal dipasang di Python 3.13/3.14 | Buat ulang environment dengan Python 3.11/3.12: `py -3.12 -m venv .venv`. |
| `Address already in use` / port 8501 sibuk | Jalankan dengan `--server.port 8502` dan buka port tersebut. |
| Halaman kosong atau error koneksi | Pastikan terminal Streamlit masih hidup, buka `http://localhost:8501` (bukan `0.0.0.0`), lalu refresh paksa `Ctrl+F5`. |
| Aplikasi terbuka tetapi tidak ada model mangga | Normal pada clone baru. Jalankan demo sintetis, latih model, atau unggah `best.pt` tepercaya sesuai langkah 8. |
| `cv2` error setelah memasang paket lain | Hapus `opencv-python-headless` dari environment, lalu pasang ulang `requirements.txt`. |
| Kamera browser tidak muncul | Gunakan `localhost` atau HTTPS, izinkan kamera di browser/OS, dan hindari iframe tanpa izin kamera. |
| Video tracking tidak dipratinjau di browser | Pasang FFmpeg dan pastikan tersedia di PATH. Tanpa FFmpeg, unduh video dan buka dengan VLC. |
| Proses lambat / kehabisan RAM | Turunkan sisi terpanjang gambar, batas frame video, dan jumlah foto per sesi. Tutup aplikasi lain saat training. |

### 12. Reset atau memperbarui instalasi

Untuk memperbaiki environment yang rusak, hapus `.venv`, lalu ulangi langkah 3–6:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Untuk mengambil pembaruan kode:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Data hasil analisis tersimpan di `output/` dan tidak terhapus saat environment dibuat ulang.

Lisensi Ultralytics adalah AGPL-3.0 / Enterprise; periksa ketentuannya bila aplikasi akan didistribusikan. Versi Ultralytics dikunci pada versi yang diuji agar API ByteTrack tetap kompatibel.

## Mulai cepat

1. Tekan **Coba data uji sintetis** pada Dashboard. Tiga foto berisi tiga bentuk bertekstur diproses menjadi sembilan deteksi dan tiga ID unik; ini pengujian buatan, bukan pengukuran kebun.
2. Periksa tampilan sebelum/sesudah, tabel deteksi, dan bukti penggabungan pada **Asosiasi & ID**.
3. Untuk foto sendiri, pilih **YOLO mangga · model sendiri** (otomatis jika model hasil training tersedia), lalu unggah foto/video pada **Akuisisi & Analisis**. Foto mangga di tangan, meja, atau keranjang juga dapat diproses. Model lain tetap perlu dikonfirmasi sumbernya.
4. Tinjau ID dan penggabungan manual sebelum memasukkan data lapangan pada **Kematangan & Panen**.
5. Masukkan hitungan manual / anotasi pada **Evaluasi**. Unduh hasil di **Riwayat & Ekspor**.

Untuk mencoba tracking tanpa model mangga, buka **Akuisisi & Analisis** dalam mode sintetis dan tekan **Demo video sintetis · ByteTrack**. Video uji berisi 24 frame dengan tiga bentuk yang sama, bukan pengambilan gambar kebun.

Satu sesi berasal dari satu pohon atau satu kelompok buah setelah dipetik. Jangan mencampur kelompok berbeda bila ingin menghubungkan buah yang sama antarfoto. Sistem tidak mengekstrapolasi jumlah mangga yang tidak pernah terlihat. Tidak ada bbox berarti “tidak terdeteksi pada threshold ini”, bukan bukti pasti bahwa gambar bukan mangga.

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
