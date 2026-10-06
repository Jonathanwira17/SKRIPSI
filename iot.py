import numpy as np
import networkx as nx
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# --- 1. INISIALISASI SASTRAWI (Stemmer & Stopword) ---
print("[SISTEM] Memuat Library NLP (Sastrawi)...")
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Mengambil stopword bawaan Sastrawi
stopword_factory = StopWordRemoverFactory()
stopword_bawaan = stopword_factory.get_stop_words()

# MENAMBAHKAN CUSTOM STOPWORD SENDIRI
# MENAMBAHKAN CUSTOM STOPWORD SENDIRI (REVISI FINAL DOSEN)
stopword_tambahan = [
    'guna', 'cara', 'lapis', 'hari', 'sehari', 'buat', 'dasar', 'atas', 'luar', 
    'bagai', 'mungkin', 'langsung', 'layan', 'ubah', 'dan', 'lebih', 'and', 'the', 
    'of', 'in', 'to', 'for', 'with', 'is', 'on', 'as', 'by', 'that', 'this',
    'yang', 'ada', 'kembang', 'hubung', 'lalu', 'cepat', 'jadi', 'kumpul', 
    'kurang', 'baru', 'telah', 'saat', 'dari', 'bisa', 'akan', 'untuk', 'sangat',
    'hadap', 'mampu', 'tingkat', 'sesuai', 'hasil', 'besar', 'perlu',
    'butuh', 'baik', 'rupa', 'utama', 'sedia', 'dukung', 'organisasi',
    # --- TAMBAHAN NOISE TERAKHIR ---
    'bantu'
]
kumpulan_stopword_baru = stopword_bawaan + stopword_tambahan

# Memasang kamus baru ke mesin
dictionary = ArrayDictionary(kumpulan_stopword_baru)
stopword_remover = StopWordRemover(dictionary)

def normalisasi_skor(skor_dict):
    """Fungsi untuk mengubah skor ke rentang 0.0 - 1.0 (Min-Max Scaling)"""
    if not skor_dict: return {}
    max_val = max(skor_dict.values())
    min_val = min(skor_dict.values())
    if max_val == min_val: return {k: 1.0 for k in skor_dict}
    return {k: (v - min_val) / (max_val - min_val) for k, v in skor_dict.items()}

# --- 2. FUNGSI TF-IDF ---
def hitung_tfidf(dokumen_list, target_index):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dokumen_list)
    fitur = vectorizer.get_feature_names_out()
    skor_baris = tfidf_matrix.toarray()[target_index]
    skor_tfidf = {fitur[i]: skor_baris[i] for i in range(len(fitur)) if skor_baris[i] > 0}
    return normalisasi_skor(skor_tfidf)

# --- 3. FUNGSI TEXTRANK ---
def hitung_textrank(teks, window_size=2):
    kata_kata = teks.split()
    graf = nx.Graph()
    for i in range(len(kata_kata) - window_size + 1):
        window = kata_kata[i : i + window_size]
        for j in range(window_size):
            for k in range(j + 1, window_size):
                if window[j] != window[k]:
                    graf.add_edge(window[j], window[k])
    skor_textrank = nx.pagerank(graf, alpha=0.85)
    return normalisasi_skor(skor_textrank)

# --- 4. FUNGSI HIBRIDA ---
def hitung_hibrida(skor_tfidf, skor_textrank, bobot_tfidf=0.4):
    bobot_textrank = 1.0 - bobot_tfidf
    skor_hibrida = {}
    semua_kata = set(skor_tfidf.keys()).union(set(skor_textrank.keys()))
    
    for kata in semua_kata:
        nilai_tf = skor_tfidf.get(kata, 0.0)
        nilai_tr = skor_textrank.get(kata, 0.0)
        skor_akhir = (bobot_tfidf * nilai_tf) + (bobot_textrank * nilai_tr)
        skor_hibrida[kata] = skor_akhir
    return skor_hibrida

# --- 5. FUNGSI EKSTRAKSI PDF (PDF PARSER) ---
def ekstrak_teks_pdf(nama_file, halaman_awal, halaman_akhir):
    print(f"[SISTEM] Membaca e-book '{nama_file}'...")
    teks_full = ""
    try:
        with open(nama_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i in range(halaman_awal - 1, halaman_akhir):
                teks_full += reader.pages[i].extract_text() + " "
                
        # --- PERBAIKAN REGEX DI SINI ---
        # Menghapus pola: [Angka] | Internet of Thing(s)
        teks_full = re.sub(r'\d+\s*\|\s*Internet of Thing[s]*', ' ', teks_full, flags=re.IGNORECASE)
        
        # (Opsional) Jika ada pola yang nomornya di belakang, kita buang juga untuk jaga-jaga
        teks_full = re.sub(r'Internet of Thing[s]*\s*\|\s*\d+', ' ', teks_full, flags=re.IGNORECASE)
        
        teks_full = re.sub(r'BAB I\s*KOMPONEN.*', ' ', teks_full, flags=re.IGNORECASE)
        return teks_full
    except Exception as e:
        print(f"[ERROR] {e}")
        return ""
        
# ==========================================
# EKSEKUSI PROGRAM E-LEARNING
# ==========================================

# Teks Materi E-Book Bab I (Data Mentah Pertemuan 1)
# ==========================================
# EKSEKUSI PROGRAM SKRIPSI (PYTHON MURNI)
# ==========================================

# Pastikan file PDF Anda berada di dalam folder yang sama dengan skrip ini
nama_file_ebook = "bahan_IOT.pdf"  # Sesuaikan dengan nama file PDF asli Anda

# Panggil mesin penyedot PDF (Misal: Kita ingin membedah Bab 1 yang ada di Halaman 1 sampai 15)
teks_materi = ekstrak_teks_pdf(nama_file_ebook, 28, 47)

# Jika gagal baca PDF, program berhenti
if not teks_materi.strip():
    print("Materi kosong atau file PDF tidak ditemukan. Hentikan program.")
    exit()

# Memecah teks dan HANYA mengambil kalimat yang memiliki lebih dari 10 kata (Filter Noise Header/Footer)
# Tambahan: Membersihkan karakter tersembunyi (\r, \n) dan spasi ganda agar output tidak menumpuk di terminal
teks_materi_bersih = re.sub(r'\s+', ' ', teks_materi)
korpus_mentah = [kalimat.strip() for kalimat in teks_materi_bersih.split('.') if len(kalimat.split()) > 10]

print("[SISTEM] Sedang melakukan Preprocessing (Stemming dulu, baru Hapus Stopword)...")
korpus_bersih = []
for doc in korpus_mentah:
    # 1. STEMMING DULU (Ubah "pengguna" jadi "guna", "lapisan" jadi "lapis")
    doc_stemmed = stemmer.stem(doc)
    
    # 2. HAPUS STOPWORD MENGGUNAKAN NATIVE PYTHON (100% Ampuh)
    kata_kata = doc_stemmed.split()
    kata_bersih = [k for k in kata_kata if k not in kumpulan_stopword_baru]
    doc_no_stop = " ".join(kata_bersih)
    
    if doc_no_stop.strip():
        korpus_bersih.append(doc_no_stop)

# Menggabungkan kembali untuk TextRank
teks_full_bersih = " ".join(korpus_bersih)

print("[SISTEM] Menghitung bobot Hibrida (TF-IDF + TextRank)...")
# Eksekusi TF-IDF pada seluruh korpus, dan ambil rata-rata skornya (karena ini 1 materi utuh)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(korpus_bersih)
fitur = vectorizer.get_feature_names_out()
rata_rata_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
skor_tfidf_global = normalisasi_skor({fitur[i]: rata_rata_tfidf[i] for i in range(len(fitur))})

# Eksekusi TextRank
hasil_textrank = hitung_textrank(teks_full_bersih)

# Eksekusi Hibrida (40:60)
hasil_hibrida = hitung_hibrida(skor_tfidf_global, hasil_textrank, bobot_tfidf=0.4)

# --- VARIABEL PENGATUR JUMLAH SOAL (Ubah angka ini sesuai permintaan dosen) ---
JUMLAH_SOAL = 30

print("\n=================================================================")
print(f"🎯 TOP {JUMLAH_SOAL} KATA KUNCI (PERBANDINGAN TF-IDF, TEXTRANK, HIBRIDA)")
print("=================================================================")

# Sistem akan otomatis mengambil batas sesuai variabel JUMLAH_SOAL
kata_kunci_final = sorted(hasil_hibrida.items(), key=lambda x: x[1], reverse=True)[:JUMLAH_SOAL]

# Mencetak Header Tabel di Terminal agar rapi
print(f"{'KATA KUNCI':<15} | {'TF-IDF':<10} | {'TEXTRANK':<10} | {'HIBRIDA':<10}")
print("-" * 55)

for kata, skor_hibrida in kata_kunci_final:
    # Memanggil kembali nilai asli dari masing-masing algoritma
    skor_tf = skor_tfidf_global.get(kata, 0.0)
    skor_tr = hasil_textrank.get(kata, 0.0)
    
    # Mencetak nilai secara berdampingan (rata kiri/kanan)
    print(f"{kata.capitalize():<15} | {skor_tf:.4f}     | {skor_tr:.4f}     | {skor_hibrida:.4f}")

import random
import re

# Generasi Kuis Evaluasi Otomatis (Pilihan Ganda Dinamis)
print("\n=================================")
print("📝 GENERATOR KUIS OTOMATIS (PILIHAN GANDA)")
print("=================================")

# Kumpulkan top 15 kata kunci sebagai bank kata (untuk dijadikan pengecoh/opsi salah)
bank_kata = [k[0].capitalize() for k in sorted(hasil_hibrida.items(), key=lambda x: x[1], reverse=True)[:15]]

# --- SOLUSI: Memori untuk mencatat kalimat yang sudah dipakai ---
kalimat_terpakai = set()

# Siapkan variabel penampung teks untuk diekspor
teks_ekspor = "=================================\n"
teks_ekspor += "📝 HASIL GENERATOR KUIS OTOMATIS\n"
teks_ekspor += "=================================\n\n"

for urutan, (kata, _) in enumerate(kata_kunci_final, start=1):
    kata_target = kata.lower()
    kalimat_soal = ""
    
    # Cari kalimat asli di dokumen yang mengandung kata kunci tersebut
    for kalimat in korpus_mentah:
        # CEK: Apakah kalimat ini BELUM PERNAH dipakai untuk soal sebelumnya?
        if kalimat not in kalimat_terpakai:
            # Gunakan regex (\w*) agar mesin mentoleransi adanya akhiran kata
            if re.search(rf'\b{kata_target}\w*\b', kalimat, re.IGNORECASE):
                # Lubangi kata kunci di dalam kalimat menjadi titik-titik
                kalimat_soal = re.sub(rf'\b{kata_target}\w*\b', '.......', kalimat, flags=re.IGNORECASE)
                
                # CATAT: Masukkan kalimat ini ke memori agar tidak dipakai lagi
                kalimat_terpakai.add(kalimat)
                break
                
    if kalimat_soal:
        # Menyiapkan kerangka teks soal
        soal_cetak = f"Soal {urutan}: {kalimat_soal}\n"
        print(f"Soal {urutan}: {kalimat_soal}")
        
        # Buat opsi jawaban (A, B, C, D)
        jawaban_benar = kata.capitalize()
        opsi_salah = [k for k in bank_kata if k.lower() != kata_target]
        
        # Ambil 3 pengecoh secara acak
        pengecoh = random.sample(opsi_salah, min(3, len(opsi_salah)))
        
        # Gabungkan kunci jawaban dengan pengecoh, lalu acak urutannya
        pilihan_ganda = pengecoh + [jawaban_benar]
        random.shuffle(pilihan_ganda)
        
        # Cetak opsi A, B, C, D ke terminal sekaligus menyimpannya ke variabel ekspor
        huruf = ['A', 'B', 'C', 'D']
        for i in range(len(pilihan_ganda)):
            opsi_cetak = f"   {huruf[i]}. {pilihan_ganda[i]}\n"
            print(f"   {huruf[i]}. {pilihan_ganda[i]}")
            soal_cetak += opsi_cetak
            
        kunci_cetak = f"   (Kunci Jawaban: {jawaban_benar})\n\n"
        print(f"   (Kunci Jawaban: {jawaban_benar})\n")
        soal_cetak += kunci_cetak
        
        # Masukkan 1 soal utuh ini ke penampung utama
        teks_ekspor += soal_cetak
    else:
        # Cadangan jika kata tidak bisa dilubangi
        soal_cadangan = f"Soal {urutan}: Jelaskan secara singkat konsep dari {kata.capitalize()}!\n\n"
        print(f"Soal {urutan}: Jelaskan secara singkat konsep dari {kata.capitalize()}!\n")
        teks_ekspor += soal_cadangan

# --- FITUR EKSPOR OTOMATIS KE TXT ---
nama_file_hasil = "Hasil_Kuis_IoT.txt"
with open(nama_file_hasil, "w", encoding="utf-8") as file_txt:
    file_txt.write(teks_ekspor)

print("=================================")
print(f"✅ [SUKSES] Seluruh soal kuis berhasil diekspor dan disimpan ke dalam file: {nama_file_hasil}")
print("=================================\n")

# --- 6. MESIN EVALUATOR (MENGHITUNG PRECISION, RECALL, F1-SCORE) ---
print("\n=================================")
print("📊 EVALUASI AKURASI SISTEM (NLP METRICS)")
print("=================================")

# SIMULASI Data Ground Truth dari Pakar
ground_truth_pakar = [
    "iot", "industri", "society", "sensor", "aktuator", "perangkat", "edge", 
    "mikrokontroler", "mikroprosesor", "protokol", "komunikasi", "gateway", 
    "cloud", "big", "data", "machine", "learning", "kecerdasan", "buatan", 
    "ai", "rfid", "jaringan", "enkripsi", "autentikasi", "otomatisasi", 
    "real-time", "smart", "cities", "analitik"
]

# ADIL: Kita wajib melakukan stemming pada kata pakar agar sama persis formatnya dengan sistem
ground_truth_pakar_stemmed = [stemmer.stem(kata) for kata in ground_truth_pakar]

# Ambil kata kunci dari sistem
kata_kunci_sistem = [kata[0].lower() for kata in kata_kunci_final]

# Hitung True Positives (TP)
true_positives = set(kata_kunci_sistem).intersection(set(ground_truth_pakar_stemmed))
tp = len(true_positives)

# Hitung False Positives (FP)
fp = len(kata_kunci_sistem) - tp

# Hitung False Negatives (FN)
fn = len(ground_truth_pakar_stemmed) - tp

# Menghitung Precision, Recall, dan F1-Score (Pakai try-except untuk cegah error bagi nol)
try:
    precision = tp / (tp + fp)
except ZeroDivisionError:
    precision = 0.0

try:
    recall = tp / (tp + fn)
except ZeroDivisionError:
    recall = 0.0

try:
    f1_score = 2 * (precision * recall) / (precision + recall)
except ZeroDivisionError:
    f1_score = 0.0

# Cetak Laporan Evaluasi ke Terminal
print(f"Kata Kunci Sistem : {kata_kunci_sistem}")
print(f"Ground Truth Pakar: {ground_truth_pakar}")
print(f"Irisan (Tebakan Benar) : {list(true_positives)}\n")

print("=================================")
print(f"EVALUASI AKURASI (HIBRIDA VS PAKAR)")
print("=================================")
print(f"True Positives (TP) : {tp} kata")
print(f"False Positives (FP): {fp} kata")
print(f"False Negatives (FN): {fn} kata\n")

print(f"🎯 Precision : {precision * 100:.2f}%")
print(f"🎯 Recall    : {recall * 100:.2f}%")
print(f"🏆 F1-Score  : {f1_score * 100:.2f}%")
print("=================================\n")