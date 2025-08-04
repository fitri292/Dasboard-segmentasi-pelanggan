import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Segmentasi Pelanggan - KMeans", layout="wide")

# Sidebar navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "Unggah Data", "Hasil Clustering"]
)


# Session state
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
if "rfm" not in st.session_state:
    st.session_state.rfm = None
if "rfm_norm" not in st.session_state:
    st.session_state.rfm_norm = None
if "hasil_cluster" not in st.session_state:
    st.session_state.hasil_cluster = None

# Halaman Beranda
if menu == "🏠 Beranda":
    st.title("👋 Hai, Selamat Datang!")
    st.markdown("""
    Ini adalah aplikasi sederhana untuk **segmentasi pelanggan** menggunakan *K-Means Clustering*.
    
    Kamu bisa:
    - Membagi pelanggan jadi beberapa kelompok.
    - Mengetahui pelanggan setia atau yang butuh perhatian.
    - Menyimpan hasil analisis untuk strategi marketing.
    
    ---
    **Cara Cepat Pakai:**
    1. Buka halaman **Unggah Data**.
    2. Upload file transaksi.
    3. Lihat hasil di halaman **Hasil Clustering**.
    
    Yuk mulai! 🚀
    """)


# Halaman Unggah Data
elif menu == "Unggah Data":
    st.title("Unggah Data Transaksi")

    st.markdown("""
**Format file:**
- Tipe: `.csv` atau `.xlsx`
- Kolom wajib: `Tanggal`, `ID_Pelanggan`, `Total`

**Contoh:**

| Tanggal     | ID_Pelanggan | Total   |
|-------------|--------------|---------|
| 2025-01-02  | 123          | 50000   |
""")

    uploaded_file = st.file_uploader("Pilih file data transaksi", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.data_uploaded = True

        st.subheader("Pratinjau Data")
        st.dataframe(df.head())

        expected_cols = {"Tanggal", "ID_Pelanggan", "Total"}
        if not expected_cols.issubset(df.columns):
            st.error("Kolom tidak lengkap. Harus ada: Tanggal, ID_Pelanggan, Total.")
            st.stop()

        # Proses RFM
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        now = df["Tanggal"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("ID_Pelanggan").agg({
            "Tanggal": lambda x: (now - x.max()).days,
            "ID_Pelanggan": "count",
            "Total": "sum"
        }).rename(columns={
            "Tanggal": "Recency",
            "ID_Pelanggan": "Frequency",
            "Total": "Monetary"
        }).reset_index()

        scaler = MinMaxScaler()
        rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
        rfm_norm = pd.DataFrame(rfm_scaled, columns=["Recency_Norm", "Frequency_Norm", "Monetary_Norm"])

        # Simpan ke session
        st.session_state.rfm = rfm
        st.session_state.rfm_norm = rfm_norm

        st.success("Data berhasil diproses. Silakan ke halaman 'Hasil Clustering'.")

# Halaman Hasil Clustering
elif menu == "Hasil Clustering":
    st.title("Hasil Segmentasi Pelanggan")

    if not st.session_state.data_uploaded:
        st.warning("Silakan unggah data terlebih dahulu di halaman 'Unggah Data'.")
    else:
        rfm = st.session_state.rfm
        rfm_norm = st.session_state.rfm_norm

        # Slider jumlah kluster
        st.sidebar.subheader("Pengaturan Kluster")
        k = st.sidebar.slider("Jumlah Kluster", 2, 10, 3)

        # Elbow Method
        st.subheader("Metode Elbow")
        sse = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init="auto")
            km.fit(rfm_norm)
            sse.append(km.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), sse, marker="o")
        ax1.set_xlabel("Jumlah Kluster")
        ax1.set_ylabel("SSE")
        ax1.set_title("Elbow Method")
        st.pyplot(fig1)

        # Clustering dengan k
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        rfm["Cluster"] = model.fit_predict(rfm_norm)
        st.session_state.hasil_cluster = rfm

        # Visualisasi Kluster
        st.subheader("Visualisasi Kluster (Recency vs Monetary)")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="tab10", ax=ax2)
        ax2.set_title("Peta Kluster")
        st.pyplot(fig2)

        # Jumlah per kluster
        st.subheader("Jumlah Pelanggan per Kluster")
        cluster_counts = rfm["Cluster"].value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="tab10", ax=ax3)
        ax3.set_xlabel("Kluster")
        ax3.set_ylabel("Jumlah")
        st.pyplot(fig3)

        # Ringkasan & strategi
        st.subheader("Ringkasan Kluster dan Strategi")
        for cluster in sorted(rfm["Cluster"].unique()):
            c_data = rfm[rfm["Cluster"] == cluster]
            r_mean = c_data["Recency"].mean()
            f_mean = c_data["Frequency"].mean()
            m_mean = c_data["Monetary"].mean()

            # Strategi dinamis
            strategi = []
            if r_mean <= rfm["Recency"].quantile(0.3):
                strategi.append("Pelanggan aktif — beri program loyalitas.")
            elif r_mean >= rfm["Recency"].quantile(0.7):
                strategi.append("Sudah lama tidak belanja — kirim penawaran khusus.")

            if f_mean >= rfm["Frequency"].quantile(0.7):
                strategi.append("Transaksi sering — beri penghargaan atau bonus.")
            elif f_mean <= rfm["Frequency"].quantile(0.3):
                strategi.append("Jarang bertransaksi — dorong engagement.")

            if m_mean >= rfm["Monetary"].quantile(0.7):
                strategi.append("Nilai tinggi — pelanggan prioritas.")
            elif m_mean <= rfm["Monetary"].quantile(0.3):
                strategi.append("Nilai kecil — coba upselling.")

            with st.expander(f"Kluster {cluster}"):
                st.markdown(f"""
**Statistik Rata-rata:**
- Recency: {r_mean:.1f} hari
- Frequency: {f_mean:.1f} kali
- Monetary: Rp {m_mean:,.0f}

**Strategi:**
- {'\n- '.join(strategi)}
""")

        # Data lengkap
        st.subheader("Data Lengkap")
        st.dataframe(rfm)

        # Unduh hasil
        csv = rfm.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "hasil_klustering.csv", "text/csv")
