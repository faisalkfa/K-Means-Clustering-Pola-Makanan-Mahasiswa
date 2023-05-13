# Mengimpor library yang diperlukan
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
 
print("=====================================================================================================================")
print("PENERAPAN ALGORITMA K-MEANS CLUSTERING DENGAN PYTHON UNTUK MENGELOMPOKKAN POLA KONSUMSI MAKANAN DI KALANGAN MAHASISWA")
print("=====================================================================================================================")
 
# Membaca file CSV
file_name = 'Data_Survei_Pola_Makanan_Mahasiswa.csv'
df = pd.read_csv(file_name)
 
# Mengubah nama kolom
df.columns = ['Timestamp', 'Email', 'Name', 'Age', 'Gender', 'Veg_per_day', 'Fruit_per_day', 'Fast_food_per_week', 'High_cal_snacks_per_day', 'Carbs_per_day', 'Lean_protein_per_day', 'Water_8_glasses', 'Breakfast_daily']
 
# Mengkonversi data non-numerik menjadi numerik menggunakan mapping
df['Veg_per_day'] = df['Veg_per_day'].map({'Tidak pernah': 0, '1 kali sehari': 1, '2 kali sehari': 2, '3 kali atau lebih sehari': 3})
df['Fruit_per_day'] = df['Fruit_per_day'].map({'Tidak pernah': 0, '1 kali sehari': 1, '2 kali sehari': 2, '3 kali atau lebih sehari': 3})
df['Fast_food_per_week'] = df['Fast_food_per_week'].map({'Tidak pernah': 0, '1-2 kali seminggu': 1, '3-4 kali seminggu': 2, '5 kali atau lebih seminggu': 3})
df['High_cal_snacks_per_day'] = df['High_cal_snacks_per_day'].map({'Tidak pernah': 0, '1 kali sehari': 1, '2 kali sehari': 2, '3 kali atau lebih sehari': 3})
df['Carbs_per_day'] = df['Carbs_per_day'].map({'Tidak pernah': 0, '1 porsi': 1, '2 porsi': 2, '3 porsi atau lebih': 3})
df['Lean_protein_per_day'] = df['Lean_protein_per_day'].map({'Tidak pernah': 0, '1 kali sehari': 1, '2 kali sehari': 2, '3 kali atau lebih sehari': 3})
df['Water_8_glasses'] = df['Water_8_glasses'].map({'Ya': 1, 'Tidak': 0})
df['Breakfast_daily'] = df['Breakfast_daily'].map({'Ya': 1, 'Tidak': 0})
 
# Menghitung skor sehat dan tidak sehat berdasarkan kolom yang ada
df['healthy_score'] = df['Veg_per_day'] * 1 + df['Fruit_per_day'] * 1 + df['Lean_protein_per_day'] * 1 + df['Water_8_glasses'] * 1 + df['Breakfast_daily'] * 1
df['unhealthy_score'] = df['Fast_food_per_week'] * -1 + df['High_cal_snacks_per_day'] * -1 + df['Carbs_per_day'] * -1
 
# Menghitung skor kesehatan bersih
df['net_health_score'] = df['healthy_score'] + df['unhealthy_score']
 
# Menyiapkan data untuk clustering
data = df[['net_health_score']]
 
# Melakukan standardisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
 
# Mencari jumlah cluster terbaik/optimal menggunakan metode Elbow
inertia = []
max_clusters = 5
 
# Melakukan iterasi untuk mencari nilai optimal jumlah cluster
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=5)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
 
# Membuat grafik Elbow Method untuk menentukan jumlah cluster optimal
print("\n----------------------")
print("1. Grafik Elbow Method")
print("----------------------")
plt.plot(range(1, max_clusters + 1), inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
 
# Menyiapkan data untuk clustering dengan one-hot encoding
data = df[['net_health_score']]
data_encoded = pd.get_dummies(data, drop_first=True)
 
# Melakukan standardisasi data yang telah di-encode
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)
 
# Melakukan clustering menggunakan KMeans dengan jumlah cluster optimal berdasarkan grafik elbow method
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
labels = kmeans.fit_predict(data_scaled)
 
# Menghitung metrik evaluasi untuk clustering
 
# Menghitung Inertia (WCSS)
inertia = kmeans.inertia_
 
# Menghitung Silhouette Score
silhouette_avg = silhouette_score(data_scaled, labels)
 
# Menghitung Calinski-Harabasz Index (CH Index)
ch_index = calinski_harabasz_score(data_scaled, labels)
 
# Menghitung Davies-Bouldin Index (DB Index)
db_index = davies_bouldin_score(data_scaled, labels)
 
# Menambahkan label cluster ke DataFrame asli
df['Cluster'] = kmeans.labels_
 
# Menyimpan data dengan informasi cluster ke file CSV baru
df.to_csv('hasil_pengelompokan_baru_new.csv', index=False)
 
# Menghitung jumlah mahasiswa dalam setiap cluster
cluster_counts = df['Cluster'].value_counts()
 
# Menggantikan label cluster dengan nama yang lebih mudah dimengerti
cluster_names = {0: 'Pola Makanan Sehat', 1: 'Pola Makanan Kurang Sehat'}
df['Cluster'] = df['Cluster'].replace(cluster_names)
 
# Fungsi untuk mencetak ringkasan hasil pengelompokan
print("--------------------------------")
print("2. Ringkasan Hasil Pengelompokan")
print("--------------------------------\n")
def print_cluster_summary(cluster_summary):
    for i, row in cluster_summary.iterrows():
        print(f"{i}")
        print(f"Jumlah mahasiswa: {int(row['Jumlah Mahasiswa'])}\n")
        print("Rata Rata")
        col_names = {
            'Veg_per_day': 'Sayuran per hari',
            'Fruit_per_day': 'Buah per hari',
            'Fast_food_per_week': 'Makanan cepat saji per minggu',
            'High_cal_snacks_per_day': 'Camilan tinggi kalori per hari',
            'Carbs_per_day': 'Karbohidrat per hari',
            'Lean_protein_per_day': 'Protein tanpa lemak per hari',
            'Water_8_glasses': 'Kebiasaan minum air putih minimal 8 gelas per hari',
            'Breakfast_daily': 'Kebiasaan sarapan setiap hari',
            'healthy_score': 'Skor sehat',
            'unhealthy_score': 'Skor tidak sehat',
            'net_health_score': 'Skor kesehatan bersih'
        }
        for col, value in row[:-1].items():
            print(f"{col_names[col]}: {value}")
        print("")
 
# Membuat ringkasan hasil pengelompokan
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
cluster_summary = cluster_summary.applymap(lambda x: round(x, 3))
cluster_summary['Jumlah Mahasiswa'] = df['Cluster'].value_counts()
 
# Mencetak ringkasan hasil pengelompokan
print_cluster_summary(cluster_summary)
 
# Mencetak rekomendasi berdasarkan cluster dengan jumlah mahasiswa terbanyak
print("-------------------------------------------------------------")
print("3. Rekomendasi Yang Diberikan Berdasarkan Hasil Pengelompokan")
print("-------------------------------------------------------------")
most_students_cluster = cluster_summary['Jumlah Mahasiswa'].idxmax()
print(f"Hasil pengelompokkan sesuai dengan jumlah mahasiswa paling banyak, yaitu berada pada kategori '{most_students_cluster}':")
if most_students_cluster == "Pola Makanan Sehat":
    print("Mahasiswa yang memiliki pola makanan yang sehat perlu mempertahankan pola makanan tersebut dan dapat berbagi tips dan pengalaman dengan teman-teman mereka. Selain itu, mereka juga dapat membantu meningkatkan kesadaran tentang kesehatan di kalangan teman-teman mereka dan mengambil inisiatif untuk mempromosikan pola makanan yang sehat di kampus mereka. Dengan cara ini, mahasiswa dapat membantu meningkatkan kesehatan dan kesejahteraan di kalangan teman-teman mereka dan membantu memperkenalkan gaya hidup yang lebih sehat. \n")
else:
    print("Mahasiswa harus menjaga pola makanan seimbang dan teratur, berolahraga secara teratur, menjaga keseimbangan kalori dan mengikuti edukasi kesehatan. Dengan cara ini, mahasiswa dapat membantu mengatasi pola makanan yang kurang sehat dan memperoleh manfaat kesehatan yang lebih baik. Oleh karena itu, penting bagi mahasiswa untuk menjaga pola makanan yang sehat dan berolahraga secara teratur, serta mengikuti program edukasi kesehatan agar dapat membantu meningkatkan kesehatan dan kesejahteraan di kalangan mahasiswa. \n")
 
# Mencetak metrik evaluasi
print("---------------------------------------")
print("4. Hasil Uji Evaluasi / Metrik Evaluasi")
print("---------------------------------------")
print(f"Inertia (WCSS): {round(inertia, 3)}")
print(f"Silhouette Score: {round(silhouette_avg, 3)}")
print(f"Calinski-Harabasz Index (CH Index): {round(ch_index, 3)}")
print(f"Davies-Bouldin Index (DB Index): {round(db_index, 3)} \n")
 
# Mencetak penjelasan metrik evaluasi
print("Keterangan metrik evaluasi:")
print("1. Inertia (WCSS): Jarak antara anggota cluster dengan pusat cluster. Semakin rendah nilai WCSS, semakin baik kualitas pengelompokan.")
print("2. Silhouette Score: Nilai rata-rata jarak antara anggota cluster yang sama dan anggota cluster terdekat. Nilai berkisar -1 hingga 1. Semakin tinggi nilai Silhouette Score, semakin baik kualitas pengelompokan.")
print("3. Calinski-Harabasz Index (CH Index): Rasio antara dispersi antar-cluster dengan dispersi intra-cluster. Semakin tinggi nilai CH Index, semakin baik kualitas pengelompokan.")
print("4. Davies-Bouldin Index (DB Index): Rasio jarak antara pusat cluster dengan jarak rata-rata anggota cluster. Semakin rendah nilai DB Index, semakin baik kualitas pengelompokan.")
 
# Menetapkan ambang batas sebagai keoptimalan metrik evaluasi
inertia_threshold = 100
silhouette_threshold = 0.6
ch_index_threshold = 100
db_index_threshold = 0.6
 
# Menyimpulkan hasil keoptimalan metrik evaluasi berdasarkan ambang batas
print("\n-------------------------------------------------------")
print("5. Kesimpulan Dari Hasil Uji Evaluasi / Metrik Evaluasi")
print("-------------------------------------------------------")
if inertia <= inertia_threshold and silhouette_avg >= silhouette_threshold and ch_index >= ch_index_threshold and db_index <= db_index_threshold:
    print("Program dianggap optimal berdasarkan hasil metrik evaluasi tersebut.")
else:
    print("Program mungkin belum optimal dan memerlukan penyesuaian lebih lanjut berdasarkan hasil metrik evaluasi tersebut.")
