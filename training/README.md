# Training detektor mangga nyata

Tujuan: mendeteksi **mango**, baik pada pohon maupun setelah dipetik, dan mengurangi deteksi palsu pada buah lain. Seluruh anotasi positif berasal dari dataset beranotasi; tidak memakai bentuk sintetis sebagai data latih model ini. Label mutu atau tahap pertumbuhan dari sumber digabung menjadi satu kelas mango, bukan diperlakukan sebagai bukti kematangan fisiologis.

## Data dan atribusi

| Sumber | Cakupan | Lisensi / catatan |
|---|---|---|
| [MangoYOLO / Koirala et al.](https://figshare.com/articles/dataset/MangoYOLO_data_set/13450661), [distribusi AgML](https://huggingface.co/datasets/Project-AgML/mango_detection_australia) | 1.242 foto kebun, banyak foto malam dengan lampu | CC BY 4.0 dari penerbit asli |
| [Mango Growth / Kabir et al.](https://huggingface.co/datasets/Project-AgML/mango_growth_detection), DOI 10.17632/5snwpzdtzs.1 | Foto smartphone dengan buah pada berbagai ukuran/tahap | CC BY 4.0; ekspor berisi 2.000 gambar, diseleksi satu versi dari 924 nama foto asli |
| [Mango-YOLO / Hasanati et al.](https://github.com/nhasanati/Mango-YOLO) | 468 foto pascapanen, termasuk buah tunggal dan kumpulan buah | Anotasi dinyatakan untuk penelitian; sebagian gambar berasal dari Kaggle dan tetap mengikuti hak sumbernya. Jangan menganggap seluruh gambar bebas didistribusikan ulang |
| [Fruits-360 / Mihai Oltean](https://github.com/fruits-360/fruits-360-100x100) | 308 contoh non-mangga dari 11 kategori: apel, alpukat, pisang, jambu, kiwi, lemon, manggis, jeruk, pepaya, pir | CC BY-SA 4.0; gambar non-mangga memakai label kosong |

Arsip unduhan dan dataset hasil olahan disimpan lokal serta diabaikan Git. `dataset/sources/downloads.json` mencatat ukuran dan SHA-256 arsip. Revisi sumber positif dikunci dalam downloader; revisi Fruits-360 yang dipakai tercatat di `negative_sources.json`. Tidak ada kode atau checkpoint pihak ketiga dari repository dataset yang dijalankan; bobot awal berasal dari YOLO resmi.

## Pencegahan kebocoran dan penyeimbangan

1. Kebun: semua potongan/sisi pada baris akuisisi 2 masuk validasi, baris 3 masuk test, dan baris lain masuk train. Tidak membagi tile dari baris yang sama secara acak.
2. Smartphone: satu ekspor per nama foto sebelum `.rf.`, sehingga versi augmentasi tidak dianggap foto independen. Nama urut dikelompokkan per 10 sebelum pembagian deterministik.
3. Pascapanen: test dari penulis dipertahankan; validasi hanya diambil dari train. Varian dengan kelompok nama sama diarahkan ke split yang sama, dengan prioritas test lalu val.
4. Non-mangga: partisi Training sumber hanya untuk train; seluruh kelompok kategori dari Test sumber diarahkan ke val atau test, bukan membagi rotasi yang mirip antar keduanya.
5. Duplikat piksel dibuang. Foto yang sangat mirip antar split juga diperiksa dengan pHash dan kesalahan thumbnail warna; versi pada split held-out dipertahankan.
6. Data latih tiap domain dibatasi maksimal 400 foto yang dipilih deterministik, agar foto kebun malam tidak mendominasi foto smartphone dan pascapanen. Data validasi/test tidak dikurangi untuk menyeimbangkan skor.

Identitas spesimen/pohon tidak tersedia lengkap pada setiap sumber. Pengelompokan nama dan pHash mengurangi kebocoran, tetapi tidak membuktikan seluruh foto berasal dari buah/pohon berbeda. Pembatasan ini harus tetap disampaikan bersama metrik.

Dataset gabungan final pada eksperimen lokal (`dataset/mango_real_v3`):

| Domain | Train | Val | Test |
|---|---:|---:|---:|
| Kebun malam | 400 | 93 | 90 |
| Smartphone | 400 | 90 | 142 |
| Pascapanen | 286 | 40 | 141 |
| Non-mangga | 166 | 60 | 72 |
| Total | 1.252 | 283 | 445 |

Total 1.980 foto, 4.434 bbox mangga untuk train, 1.130 untuk val, 1.081 untuk test. Sebanyak 11 duplikat/kemiripan dibuang sebelum penyeimbangan. Foto train tambahan yang tidak dipilih tetap tercatat sebagai jumlah pada `omitted_for_training_balance`, bukan diklaim telah dilatih pada model gabungan.

Audit gambar dilakukan pada sampel, bukan inspeksi manual seluruh 1.980 foto. Dataset negatif terutama berlatar studio; untuk penggunaan lapangan tetap diperlukan uji tambahan terhadap tangan, daun, batu, makanan, mainan, dan benda lain dari kamera pengguna.

## Menjalankan ulang

Jalankan dari root proyek menggunakan interpreter environment. Pada Windows, ganti `python` di contoh berikut dengan `.\.venv\Scripts\python.exe` jika environment belum diaktifkan.

```bash
python -m dataset.download_sources
python -m dataset.build_real
python -m training.train --data dataset/mango_real/data.yaml --epochs 8 --imgsz 384 --freeze 10 --batch 4 --name mango_joint_warmup_safe
python -m training.train --data dataset/mango_real/data.yaml --weights output/training/mango_joint_warmup_safe/weights/best.pt --epochs 30 --imgsz 640 --freeze 0 --batch 4 --name mango_joint_finetune --evaluate
```

Folder dataset/run harus baru. Pada workspace pengembangan, data final berada di `dataset/mango_real_v3`; gunakan path itu bila menjalankan eksperimen berikutnya. Percobaan awal `mango_warmup` memakai dataset lama dan dihentikan; checkpoint tersebut **tidak dipakai** sebagai bobot awal training gabungan final untuk menjaga pemisahan evaluasi.

Tahap pertama melatih neck/head dengan backbone dibekukan. Tahap kedua membuka seluruh layer untuk fine-tuning. Pengaturan: YOLO11n resmi, seed 42, AdamW, learning rate awal 0,001 yang menurun linier hingga faktor 0,05, weight decay 0,0005, warmup 2 epoch, batch 4, early stopping patience 10, augmentasi HSV, rotasi sampai 15°, translasi, skala, flip horizontal dan mosaic. Mosaic ditutup pada 5 epoch terakhir. Konfigurasi sebenarnya selalu tersimpan di `args.yaml`; jangan menyimpulkan jumlah epoch selesai hanya dari argumen awal, karena early stopping/interupsi bisa menghentikannya lebih cepat.

PyTorch memakai GPU NVIDIA bila CUDA tersedia dan memori GPU kosong minimal 2 GiB; jika tidak, CPU dipakai. Thread dibatasi dan workers=0 untuk kompatibilitas Windows. Cache gambar di RAM dimatikan untuk membatasi memori. Variasi hardware/library dapat membuat hasil antar run tidak identik bit per bit meskipun seed tetap.

Mesin pengembangan memiliki GTX 1070 Ti, tetapi environment awal memakai PyTorch CPU. Sediakan beberapa GB RAM/virtual memory kosong untuk training. Saat RAM dan VRAM hampir penuh oleh aplikasi lain, percobaan batch 8 dengan RAM cache dan batch 4 tanpa cache sama-sama gagal sebelum epoch pertama selesai; hasil tersebut tidak dianggap model terlatih final.

Untuk GPU Pascal seperti GTX 1070 Ti, gunakan environment terpisah dan wheel CUDA 12.6 yang masih mendukung arsitektur tersebut:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

Jalankan perintah training dengan interpreter `.venv-gpu` untuk memakai CUDA. Environment aplikasi `.venv` tetap dapat digunakan untuk inferensi CPU. Versi GPU dipatok pada PyTorch 2.14.1 + cu126 dan torchvision 0.29.1 + cu126; PyTorch 2.15 menghapus dukungan biner Pascal dari matriks rilis. [Pengumuman resmi PyTorch](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).

Untuk komputer yang sedang digunakan aplikasi lain, coordinator berikut menunggu RAM dan VRAM masing-masing minimal 4 GiB, menjalankan kedua tahap (warmup 8 epoch, fine-tuning maksimal 30 epoch pada 640 px), mengevaluasi, lalu memasang model. Ia tidak menutup aplikasi lain dan tidak mengklaim training selesai ketika masih menunggu:

```powershell
.\.venv\Scripts\python.exe -u -m training.run_when_ready --python .venv-gpu/Scripts/python.exe --data dataset/mango_real_v3/data.yaml --gpu
```

Pada clone baru dengan dataset default, gunakan `--data dataset/mango_real/data.yaml`. Perintah harus tetap hidup sampai selesai. Status tersimpan di `output/training/queued_training.json` dan ditampilkan pada halaman Dataset & Model. Nama run dibuat unik; kegagalan disimpan sebagai `failed`, bukan hasil berhasil. Setelah kehabisan memori, bebaskan memori sebelum mengulang. Checkpoint per epoch dapat dilanjutkan dengan fasilitas `resume` Ultralytics.

## Evaluasi dan pemasangan model

Dengan `--evaluate`, skrip:

1. Memilih threshold confidence menggunakan F1 pada **val**, bukan test, dan membatasinya ke rentang 0,15–0,85.
2. Menghitung precision/recall serta AP50 dan AP50:95 menggunakan Ultralytics pada test set terpisah, keseluruhan dan per domain positif.
3. Menghitung ulang precision/recall pada threshold operasional yang dipilih. Angka ini dibedakan dari metrik Ultralytics yang dapat memakai threshold F1 optimalnya sendiri.
4. Menghitung berapa foto non-mangga yang menghasilkan minimal satu deteksi mangga, serta jumlah bbox palsu. Ini **false-positive image rate**, berbeda dengan precision bbox.
5. Menyimpan metrik, hash manifest, hash model, konfigurasi dan grafik; baru kemudian memasang `models/best.pt` dan `models/training_summary.json`. Model lama disalin menjadi `previous_<timestamp>.pt` sebelum diganti.

`models/training_summary.json` adalah sumber angka aktual. Jika file ini belum ada, training/evaluasi/pemasangan belum selesai. `output/training/<run>/results.csv` berisi progres per epoch; `console.log` berisi log lengkap, dan `weights/last.pt` menyimpan checkpoint. Skrip tidak mengarang skor ketika training gagal.

Folder `evaluation` juga menyimpan `test_predictions.json` dan `test_errors.csv`, sehingga bbox prediksi, ground truth, serta false positive / false negative tiap foto dapat ditinjau tanpa hanya melihat skor rata-rata.

Aplikasi mengenali model bawaan hanya jika SHA-256 bobot cocok dengan laporan. Mode YOLO mangga dan threshold hasil val dipilih otomatis. CLAHE dimatikan secara default untuk model mangga agar preprocessing sesuai evaluasi. Foto yang tidak menghasilkan bbox dilaporkan sebagai “tidak ada mangga terdeteksi pada threshold ini”, bukan bukti pasti bahwa objek bukan mangga.

## Perbaikan berikutnya berdasarkan kesalahan nyata

- Kumpulkan foto kamera pengguna: mangga hijau/kuning/kemerahan, di tangan/meja/keranjang, bertumpuk, terpotong, jauh/dekat, pagi/siang/malam, serta objek mirip mangga. Buat test pribadi dari sesi/pohon/buah yang tidak ada pada train.
- Anotasi **semua** mangga yang terlihat. Foto non-mangga diberi label kosong; jangan memasukkan mangga yang tidak dianotasi sebagai negatif.
- Periksa contoh false positive dan false negative dari model, perbaiki anotasi, lalu tambah data sulit tersebut ke train. Jangan memindahkan foto test yang gagal ke train dan tetap memakai skor test lama sebagai bukti.
- Ulangi val/test setelah perubahan dan bandingkan precision, recall, AP50:95, serta false-positive rate. Banyak epoch tanpa data beragam tidak menjamin generalisasi.
- Simpan versi model dan data. Untuk percobaan GPU dengan gambar lebih besar/epoch lebih banyak, gunakan nama run baru dan test yang tetap tertutup selama pemilihan model.

Model deteksi ini tidak membuktikan kematangan fisiologis, tidak menjamin identitas buah lintas sudut, dan tidak dapat melihat buah yang sepenuhnya tertutup. Batas tersebut berbeda dari kemampuan mendeteksi mango pada satu foto.
