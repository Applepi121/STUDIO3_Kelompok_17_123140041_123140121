# STUDIO3_123140041_123140121

# Studio Proyek 3: Fine-Tuning Transformer Pretrained untuk Klasifikasi Sentimen (SmSA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14rK0rM3ozPZGoYe4OfuhpnC4sc87EL9g?usp=sharing)

Repositori ini berisi kode eksperimen, matriks evaluasi, dan analisis untuk **Studio Proyek 3** pada mata kuliah **Pembelajaran Mendalam (IF25-40401)**, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA).

Proyek ini mengevaluasi perbandingan kinerja, efisiensi komputasi, dan waktu inferensi dari tiga arsitektur *Transformer pretrained* dengan tiga strategi *fine-tuning* yang berbeda pada tugas klasifikasi sentimen teks berbahasa Indonesia (**SmSA - IndoNLU**)[cite: 1].

---

## 📋 Deskripsi Eksperimen

### 1. Dataset & Prapemrosesan
* **Dataset:** **SmSA** (*Document-level Sentiment Analysis*) dari tolok ukur **IndoNLU** dengan 3 kelas sentimen (`negative`, `neutral`, `positive`)[cite: 1].
* **Pembersihan Data:** Menghapus baris dengan teks duplikat maupun teks identik yang memiliki label bertentangan, serta memastikan tidak ada kebocoran data (*data leakage*) antara himpunan latih, validasi, dan uji[cite: 1].
* **Pembagian Data:**
  * **Data Latih:** Subset *stratified* sebanyak **5.000 kalimat** (dari 10.933 data bersih) dengan distribusi kelas proporsional (`positive`: 58.4%, `negative`: 31.2%, `neutral`: 10.4%) agar seluruh matriks eksperimen dapat dijalankan secara efisien pada GPU Tesla T4[cite: 1].
  * **Data Validasi:** **1.246 kalimat** (dipakai utuh)[cite: 1].
  * **Data Uji:** **500 kalimat** (dipakai utuh)[cite: 1].

### 2. Matriks Eksperimen (3 × 3 × 3 = 27 Run)
Eksperimen menguji kombinasi penuh dari:
1. **3 Model Backbone:**
   * **mBERT** (`bert-base-multilingual-cased`)[cite: 1]
   * **DistilmBERT** (`distilbert-base-multilingual-cased`)[cite: 1]
   * **IndoBERT** (`indobenchmark/indobert-base-p1`)[cite: 1]
2. **3 Strategi Fine-Tuning:**
   * **`full` (Full Fine-Tuning):** Seluruh parameter *backbone* dan *classification head* dilatih (`LR = 3e-5`)[cite: 1].
   * **`partial` (Partial Freezing):** Lapisan *embedding* dan separuh lapisan Transformer awal dibekukan (6 dari 12 lapisan pada BERT; 3 dari 6 lapisan pada DistilBERT), sementara separuh lapisan akhir dan *head* dilatih (`LR = 3e-5`)[cite: 1].
   * **`probing` (Linear Probing):** Seluruh *backbone* dibekukan dan hanya *classification head* yang dilatih (`LR = 3e-3`)[cite: 1].
3. **3 Random Seed:** `42`, `123`, dan `2026` (seluruh hasil dilaporkan dalam format `rata-rata ± simpangan baku`)[cite: 1].

### 3. Protokol Pelatihan & Evaluasi
* **Konfigurasi Terkunci:** `max_length=128`, `batch_size=32`, `epochs=3`, optimizer `AdamW` (`weight_decay=0.01`), *linear warmup* 10%, presisi campuran `fp16`, serta *gradient clipping* (`max_norm=1.0`)[cite: 1].
* **Aturan Evaluasi Ketat:** Himpunan validasi digunakan **hanya** untuk memilih *checkpoint* epoch terbaik (berdasarkan **F1 Makro** validasi tertinggi)[cite: 1]. Bobot epoch terbaik dipulihkan sebelum mengevaluasi himpunan uji **tepat satu kali** per *run*[cite: 1].
* **Baseline Pembanding:** `TF-IDF (1-2 gram) + Logistic Regression` (pada data latih penuh maupun subset 5.000) serta acuan tebakan acak (F1 makro ≈ 0,33)[cite: 1].

---

## 📊 Ringkasan Hasil Utama

Pengujian dijalankan pada akselerator **NVIDIA Tesla T4** (Kode Konfigurasi: `909e73`)[cite: 1]:

| Backbone | Strategi | Akurasi Uji (%) | F1 Makro Uji (%) | Waktu Latih (s) | Inferensi (ms/kal.) | Memori Puncak (GB) | Param. Dilatih (Jt) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndoBERT** | `full` | **90,93 ± 0,92** | **87,85 ± 1,70** | 75,2 | 0,867 | 3,48 | 124,444 |
| **IndoBERT** | `partial` | **90,93 ± 0,58** | **87,81 ± 0,78** | 42,1 | 0,916 | 1,79 | 43,120 |
| **IndoBERT** | `probing` | 79,93 ± 0,31 | 75,04 ± 0,32 | 19,3 | 0,897 | 0,57 | 0,002 |
| **mBERT** | `full` | 82,87 ± 1,75 | 77,80 ± 1,92 | 99,6 | 1,061 | 4,57 | 177,856 |
| **mBERT** | `partial` | 72,07 ± 1,10 | 68,19 ± 1,32 | 51,9 | 1,083 | 2,15 | 43,120 |
| **mBERT** | `probing` | 52,07 ± 0,23 | 43,93 ± 0,31 | 25,0 | 1,114 | 0,79 | 0,002 |
| **DistilmBERT** | `full` | 77,73 ± 1,03 | 74,01 ± 1,01 | 57,3 | **0,673** | 3,01 | 135,327 |
| **DistilmBERT** | `partial` | 65,67 ± 1,45 | 61,26 ± 1,29 | 25,3 | **0,591** | 1,28 | 21,857 |
| **DistilmBERT** | `probing` | 57,80 ± 0,72 | 51,84 ± 1,32 | **12,9** | **0,591** | 0,63 | 0,593 |
| *Baseline* | *TF-IDF + LogReg (Full 10.933)* | *81,00* | *75,00* | *3,39* | *0,122* | — | — |
| *Baseline* | *TF-IDF + LogReg (Subset 5.000)* | *75,00* | *67,79* | *0,94* | *0,063* | — | — |

### Temuan Kunci & Verifikasi Hipotesis
1. **Keunggulan Model Monolingual (H1 Terbukti):** **IndoBERT** mengungguli seluruh model multibahasa di setiap strategi pelatihan[cite: 1]. Bahkan strategi *linear probing* pada IndoBERT (`75,04%`) sudah mampu menyamai *baseline* `TF-IDF + LogReg` yang dilatih pada data penuh (`75,00%`)[cite: 1].
2. **Trade-off Kecepatan vs. Akurasi pada DistilmBERT (H2):** Pada pengukuran waktu inferensi satu sesi (5 pengulangan), **DistilmBERT** mencatatkan kecepatan **0,554 ± 0,015 ms/kalimat**, atau **~1,91× lebih cepat** dibandingkan **mBERT** (**1,056 ± 0,009 ms/kalimat**)[cite: 1]. Namun, penurunan F1 makro pada `full fine-tuning` tercatat sebesar **3,79 poin** (`77,80%` ke `74,01%`)[cite: 1].
3. **Efisiensi Partial Freezing pada IndoBERT (H3):** Pada IndoBERT, strategi `partial freezing` menghasilkan F1 makro **87,81 ± 0,78%**, nyaris identik dengan `full fine-tuning` (**87,85 ± 1,70%**; selisih hanya 0,03 poin dan tidak bermakna secara statistik dibanding simpangan baku antar-seed), sekaligus menghemat **44,0% waktu latih** dan **48,6% memori puncak GPU**[cite: 1].

---

## 📁 Struktur Berkas & Luaran

Setelah notebook dijalankan dalam `MODE = "penuh"`, skrip secara otomatis menghasilkan berkas CSV dan grafik berikut[cite: 1]:

* `Studio_Proyek_3_SmSA.ipynb` — Notebook utama eksperimen (Google Colab)[cite: 1].
* `hasil_smsa_909e73.csv` — Log mentah dari seluruh 27 *run* (mendukung fitur *resume* otomatis jika sesi terputus)[cite: 1].
* `rekap_agregat_smsa_909e73.csv` — Agregasi metrik rata-rata dan simpangan baku per kombinasi `(backbone, strategi)`[cite: 1].
* `tabel_laporan_909e73.csv` — Tabel ringkasan terformat yang siap digunakan pada laporan[cite: 1].
* `baseline_tfidf_909e73.csv` — Hasil evaluasi model *baseline* `TF-IDF + Logistic Regression`[cite: 1].
* `sensitivitas_lr_probing_909e73.csv` — Hasil uji diagnostik *learning rate* (`1e-3`, `3e-3`, `5e-3`) untuk strategi *linear probing* pada `mBERT` dan `DistilmBERT`[cite: 1].
* `inferensi_satu_sesi_909e73.csv` — Hasil pengukuran ulang waktu inferensi ketiga *backbone* dalam satu sesi GPU yang sama[cite: 1].

---

## 🚀 Cara Menjalankan

### Melalui Google Colab (Direkomendasikan)
1. Klik lencana **Open In Colab** di bagian atas README ini[cite: 1].
2. Pastikan runtime menggunakan akselerator GPU (**Runtime > Change runtime type > T4 GPU**).
3. Atur variabel `MODE` pada sel konfigurasi:
   * `MODE = "uji"` — Uji cepat alur kode (1 *backbone*, 1 *seed*, 1 *epoch*, 500 data latih)[cite: 1].
   * `MODE = "penuh"` — Menjalankan matriks eksperimen lengkap (27 *run* + *baseline* + uji sensitivitas LR + pengukuran inferensi + visualisasi)[cite: 1].
4. Jalankan seluruh sel dari atas ke bawah (**Runtime > Run all**)[cite: 1].

### Menjalankan Secara Lokal
```bash
# 1. Clone repositori
git clone [https://github.com/](https://github.com/)<username>/<nama-repo>.git
cd <nama-repo>

# 2. Instal dependensi utama
pip install torch transformers scikit-learn pandas matplotlib jupyter

# 3. Buka notebook
jupyter notebook
