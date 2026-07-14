# Dokumentasi Tampilan Hasil

Folder ini digunakan untuk menyimpan bukti visual setelah notebook selesai dijalankan.

## File yang Disarankan

1. `hasil_deteksi.jpg` — minimal satu gambar dengan bounding box, label, dan confidence.
2. `confusion_matrix_normalized.png` — confusion matrix hasil test.
3. `results.png` — kurva training dan validation.
4. `PR_curve.png` — kurva precision–recall.
5. `F1_curve.png` — kurva F1 terhadap confidence.

## Tabel Hasil Evaluasi

Isi setelah training selesai.

| Metrik | Nilai |
|---|---:|
| Precision | ... |
| Recall | ... |
| mAP50 | ... |
| mAP50–95 | ... |

## Analisis Singkat

Tuliskan:

- kelas yang paling mudah dideteksi;
- kelas yang paling sering tertukar;
- pengaruh pencahayaan, jarak, atau sudut wajah;
- contoh false positive dan false negative;
- saran peningkatan dataset atau parameter training.

> Jangan mengisi nilai metrik dengan perkiraan. Salin nilai aktual dari output notebook.
