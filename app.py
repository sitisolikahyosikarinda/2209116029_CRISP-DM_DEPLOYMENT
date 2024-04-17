import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.cluster import KMeans

st.set_option('deprecation.showPyplotGlobalUse', False)

# Data untuk melatih model KMeans
train_data = pd.read_csv('BankLoans (2).csv')

# Membuat model KMeans dengan 3 klaster
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_data)

# Simpan model ke dalam file 'model.pkl'
joblib.dump(kmeans, 'model.pkl')

# Data contoh
debtinc = 5
default = 0 
employ = 1 
AgeCategory = 'mapan'
age = 30 
ed = 2 
income = 5000 

# AWAL INTRODUCE
from sklearn.cluster import KMeans

selected_page = st.sidebar.selectbox(
    'Select Page',
    ['Introducing','Data Distribution','Relationship Analysis','Composition & Comparison','Clustering & Aanalysis']
)

st.title('Credit Risk Analysis for extending Bank Loans')
st.image('bank.jpg', caption='Bank Loans', use_column_width=True)

URL = 'BankLoans (2).csv'
df = pd.read_csv(URL)

if selected_page == 'Introducing':
    st.subheader("Bank Loans")
    st.write("""
        Penilaian kredit merupakan salah satu aplikasi prediksi yang paling "klasik".
        pemodelan, untuk memprediksi apakah kredit yang diberikan kepada pemohon kemungkinan besar akan terjadi atau tidak
        menimbulkan keuntungan atau kerugian bagi lembaga pemberi pinjaman. Ada banyak variasi
        dan kompleksitas mengenai bagaimana tepatnya kredit diberikan kepada individu,
        bisnis, dan organisasi lain untuk berbagai tujuan (pembelian
        peralatan, real estate, barang konsumen, dan sebagainya), dan kegunaannya bermacam-macam
        metode kredit (kartu kredit, pinjaman, rencana pembayaran tertunda). Namun dalam semua kasus, a
        pemberi pinjaman memberikan uang kepada individu atau lembaga dan mengharapkan untuk dibayar
        kembali ke masa lalu dengan bunga yang sepadan dengan risiko gagal bayar.

    """)
    st.markdown("[Sumber : Kaggle.com](https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans)")
    st.title("Kategori Umur pemohon ")
    st.write("""
        Umur pemohon yang diklompokkan berdasarkan kriteria yang telah diprediksi dari pemohon gagal bayar. Pengelompokkan umur dibagi menjadi 2 yaitu 
             mapan dan mampu. Pengglompokkan ini bertujuan untuk memprediksi umur pemohon agar tidak terjadi gagal bayar dimasadepan. """)

    def displa_AgrCategory(AgeCategory):
        if AgeCategory == "Mapan":
            st.subheader("Mapan Age Category")
            mapan_info = [
                "- Digolongkan dari umur 35 - 60 tahun, dimana umur ini dianggap sudah memiliki penghasilan yang jelas dan tertata",
                "- Untuk kategori mapan riwayat edukasi berada di level teratas",
                "- Penghasilan dari kategori mapan sudah stabil unyuk mengajukan permohonan "
            ]
            st.markdown("\n".join(mapan_info))
        elif AgeCategory == "Mampu":
            st.subheader("Mampu Age Category")
            mampu_info = [
                "- Digolongkan umur dibawah 35  tahun, dimana umur ini dianggap sudah memiliki penghasilan tapi masih belum tertata",
                "- Untuk kategori mampu riwayat edukasi masih di level standar",
                "- Penghasilan dari kategori mampu sudah dianggap bisa unyuk mengajukan permohonan "
            ]
            st.markdown("\n".join(mampu_info))

    AgeCategories = ["Mapan", "Mampu"]
    AgeCategory = st.selectbox("Age Category", AgeCategories)

    displa_AgrCategory(AgeCategory)

    st.title("Bank Loans")
    st.write("Berikut adalah data yang sudah di cleaning ")
    st.write(df)

    st.subheader('Past Default Rate')
    default_counts = df['default'].value_counts()
    st.text(f'Tingkat Gagal Bayar Sebelumnya: {default_counts.values[1] / sum(default_counts):.2%}')

# AWAL DATA DISTRIBUTION
elif selected_page == "Data Distribution":
    st.subheader("Data Distribution Section")
    umur_not_null = df['age'].dropna()
    st.title("Distribusi Umur Nasabah yang Mengajukan Pinjaman")
    plt.figure(figsize=(10, 6))
    sns.histplot(umur_not_null, bins=20, kde=True)
    plt.xlabel('Umur')
    plt.ylabel('Frekuensi')
    plt.title('Distribusi Umur Nasabah yang Mengajukan Pinjaman')
    st.pyplot()
    st.write("Grafik ini menunjukkan bahwa nasabah yang mengajukan pinjaman paling banyak berada pada rentang usia 30-35 tahun. Frekuensi nasabah menurun secara bertahap pada rentang usia yang lebih muda dan lebih tua.")
# AKHIR DATA DISTRIBUTION

# AWAL RELATIONSHIP
elif selected_page == "Relationship Analysis":
    st.subheader("Relationship Analysis Section")
    correlation_matrix = df.corr()
    st.title("Heatmap Korelasi Fitur")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()
    st.write(" Fitur 'age' (usia) dan 'income' (pendapatan) memiliki korelasi yang kuat. Hal ini menunjukkan bahwa ada hubungan positif antara usia dan pendapatan. Semakin tua usia nasabah, semakin tinggi kemungkinan pendapatannya.")
    st.write("Fitur 'employ' (pekerjaan) dan 'income' (pendapatan) juga memiliki korelasi yang kuat. Hal ini menunjukkan bahwa ada hubungan positif antara memiliki pekerjaan dan pendapatan. Nasabah yang memiliki pekerjaan lebih cenderung memiliki pendapatan yang lebih tinggi.")
    st.write("Fitur 'debtinc' (rasio utang terhadap pendapatan) dan 'income' (pendapatan) memiliki korelasi yang kuat. Hal ini menunjukkan bahwa ada hubungan positif antara pendapatan dan rasio utang terhadap pendapatan. Semakin tinggi pendapatan nasabah, semakin tinggi kemungkinan rasio utang terhadap pendapatannya.")
    st.write("Fitur 'age' (usia) dan 'default' (kemacetan) memiliki korelasi yang lemah. Hal ini menunjukkan bahwa tidak ada hubungan yang kuat antara usia dan kemungkinan macet. Usia nasabah tidak selalu menjadi indikator yang baik dari risiko kreditnya.")
   
# AKHIR RELATIONSHIP

# AWAL COMPARISION
elif selected_page == "Composition & Comparison":
    st.subheader("Composition & Comparison Section nasabah yang mengajukan pinjaman berdasarkan tingkat pendidikan")
    ed_counts = df['ed'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(ed_counts, labels=ed_counts.index, autopct='%1.1f%%', startangle=360)
    ax.axis('equal')  # Memastikan lingkaran pie terlihat sebagai lingkaran
    st.pyplot(fig)
    st.write("Grafik ini menunjukkan bahwa tingkat pendidikan pertama/1 adalah yang paling umum di antara nasabah yang mengajukan pinjaman. Persentase nasabah menurun secara bertahap pada tingkat pendidikan yang lebih tinggi.")
 
 # Visualisasi Histogram untuk distribusi rasio utang terhadap pendapatan
    plt.figure(figsize=(10, 6))
    plt.hist(df['debtinc'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribusi Rasio Utang terhadap Pendapatan')
    plt.xlabel('Rasio Utang terhadap Pendapatan')
    plt.ylabel('Jumlah Nasabah')
    st.pyplot()

    st.write("Histogram di atas menampilkan distribusi jumlah nasabah yang mengajukan pinjaman berdasarkan rasio utang terhadap pendapatan.")
# AKHIR COMPARISION


# AWAL ANALYSIS 
elif selected_page == "Clustering & Aanalysis":
    st.subheader("Clustering & Analysis Section")
    st.write("Untuk menentukan jumlah klaster yang optimal dalam analisis klaster, digunakan metode elbow. Metode ini memungkinkan untuk menemukan titik kluster yang dilihat pada saat penambahan klaster tidak lagi memberikan penurunan inersia yang signifikan.")
    
    # Elbow Method
    inertia_values = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_data)
        inertia_values.append(kmeans.inertia_)
    
    st.subheader('Elbow Method')
    st.write("""
        Pada grafik, terdapat kurva yang menunjukkan nilai inertia untuk
        setiap nilai k. Kurva ini menunjukkan siku (elbow) di sekitar k=3.
        Berdasarkan grafik Elbow Method, nilai k yang optimal untuk model
        clustering adalah 3. Hal ini menunjukkan bahwa data dapat dikelompokkan 
        secara optimal menjadi 3 cluster. """)
    
    fig, ax = plt.subplots()
    ax.plot(k_values, inertia_values, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')

    # Plot Elbow
    st.pyplot(fig)
    st.caption("Untuk mengetahui jumlah klaster yang optimal, perhatikan elbow point yang menunjukkan penurunan inersia yang signifikan atau mulai melambat hingga membentuk siku. Pada grafik Elbow di atas, terlihat bahwa pada kluster 3 mulai terjadi penurunan inersia, sehingga kluster 3 akan digunakan pada analisis ini.")

    # Slider pilih cluster
    clust = st.slider("Pilih jumlah cluster:", 2, 10, 3, 1) # rangenya 2 hingga 10 yang dimulai dari 3 dan selisih tiap range adalah 1
    
    # KMeans 
    kmeans = KMeans(n_clusters=clust, random_state=42)
    kmeans.fit(train_data)
    kmeans_clust = kmeans.predict(train_data)

    # Create DataFrame
    kmeans_col = pd.DataFrame(kmeans_clust, columns=["kmeans_cluster"])
    combined_data_assoc = pd.concat([train_data, kmeans_col], axis=1) 

    # Scatter Plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(combined_data_assoc['age'], combined_data_assoc['income'], c=combined_data_assoc['kmeans_cluster'], cmap='plasma')
    plt.title('Scatter K-Means')
    plt.xlabel('Age')
    plt.ylabel('Income')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)
    st.caption("Gambar di atas merupakan penggambaran scatter plot dari algoritma K-Means.")

    # Hasil clustering
    st.write("Berikut hasil dari pengelompokan K-Means:")
    st.write(train_data.merge(combined_data_assoc, left_index=True, right_index=True))

    st.write("### Kesimpulan")
    st.write(""" **Insight:**
    - Usia pemohon tidak secara signifikan memengaruhi pendapatan. Namun, terlihat bahwa ada kecenderungan pada peningkatan pendapatan seiring bertambahnya usia pemohon.
    - Meskipun tidak ada korelasi yang signifikan antara usia pemohon dan pendapatan, terdapat beberapa cluster yang mungkin menunjukkan pola yang berbeda dalam hubungan tersebut.
    - Dari analisis klaster, diidentifikasi bahwa jumlah klaster optimal adalah 3, yang dapat digunakan untuk segmentasi pemohon lebih lanjut dan pengembangan strategi dalam memberi pinjaman yang lebih efektif.""")

    st.write(""" **Aksi yang Dapat Diambil:**
    - Untuk meningkatkan penawaran pinjaman kepada berbagai segmen pemohon, bank dapat menggunakan segmentasi klaster ini untuk menyesuaikan strategi pemasaran dan penawaran kredit.
    - Melalui analisis lebih lanjut pada setiap klaster, bank dapat mengidentifikasi kebutuhan dan preferensi khusus dari masing-masing segmen pemohon, sehingga meningkatkan akurasi dalam menentukan kelayakan pemberian pinjaman.""")

# AKHIR ANALYSIS
