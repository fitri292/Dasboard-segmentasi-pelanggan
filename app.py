import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from io import BytesIO

# --- Pastikan openpyxl terinstal ---
# Jalankan di terminal Anda: pip install openpyxl
# ------------------------------------

# Layout halaman tetap "wide" untuk fleksibilitas
st.set_page_config(page_title="Segmentasi Pelanggan", layout="wide")

# ------------------------------ CSS Kustom ------------------------------
st.markdown("""
<style>
    /* Ukuran font dasar */
    html, body, [class*="css"] { 
        font-size: 18px !important; 
    }
    h1 { 
        font-size: 40px !important; 
        font-weight: bold !important; 
        margin-top: -35px !important; /* Menaikkan posisi judul */
        /* --- PERUBAHAN 1: Menambah jarak di bawah judul utama --- */
        margin-bottom: 30px !important; 
    }
    h2 { 
        font-size: 28px !important; 
        font-weight: bold !important; 
    }
    h3 { 
        font-size: 24px !important; 
        font-weight: bold !important; 
    }

    /* --- PERBAIKAN CSS DI SINI: Selector lebih spesifik untuk judul expander --- */
    /* Menargetkan elemen paragraf (p) di dalam judul (summary) expander */
    div[data-testid="stExpander"] summary p {
        font-size: 24px !important; /* Ubah ukuran sesuai selera, misal: 26px */
        font-weight: bold !important; /* Membuat teks menjadi tebal */
    }
    
    /* Styling untuk tabel DataFrame bawaan Streamlit (Data Awal, RFM, dll.) */
    .stDataFrame div, .stDataFrame table { 
        font-size: 22px !important;
    }
    .stDataFrame th, .stDataFrame td { 
        text-align: center !important;
        padding: 18px !important;
    }
    /* Styling untuk tabel strategi kustom */
    .strategy-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 18px; 
        text-align: left;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
    .strategy-table th, .strategy-table td {
        border: 1px solid #dddddd;
        padding: 16px 20px; 
        vertical-align: top;
    }
    .strategy-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
    }
    .strategy-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .strategy-table ul {
        padding-left: 20px;
        margin: 0;
    }
    .strategy-table li {
        margin-bottom: 8px; 
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Fungsi-Fungsi ------------------------------
def deteksi_kolom(data):
    """Mendeteksi kolom yang relevan (Tanggal, ID_Pelanggan, Total, Nama, No Telpon) secara otomatis."""
    kolom = {}
    required_keys = {'Tanggal', 'ID_Pelanggan', 'Total', 'Nama'}
    
    for c in data.columns:
        c_low = c.lower()
        if 'tgl' in c_low or 'date' in c_low or 'tanggal' in c_low:
            kolom['Tanggal'] = c
        elif ('id' in c_low and 'pelanggan' in c_low) or 'customerid' in c_low or ('id' in c_low and 'cust' in c_low):
            kolom['ID_Pelanggan'] = c
        elif 'total' in c_low or 'amount' in c_low or 'nilai' in c_low:
            kolom['Total'] = c
        elif 'nama' in c_low or ('customer' in c_low and 'name' in c_low):
            kolom['Nama'] = c
        elif 'telp' in c_low or 'phone' in c_low or 'nomor' in c_low:
            kolom['No Telpon'] = c
            
    if not required_keys.issubset(kolom.keys()):
        return None
    
    return kolom

def hitung_rfm(data):
    """Menghitung nilai Recency, Frequency, dan Monetary."""
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    snapshot_date = data['Tanggal'].max() 
    rfm = data.groupby('ID_Pelanggan').agg({
        'Tanggal': lambda x: (snapshot_date - x.max()).days,
        'ID_Pelanggan': 'count',
        'Total': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def normalisasi_rfm(rfm):
    """Normalisasi data RFM menggunakan MinMaxScaler."""
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(rfm), columns=rfm.columns, index=rfm.index)

def lakukan_clustering(rfm_norm, n_clusters=3):
    """Melakukan clustering menggunakan KMeans."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_norm['Cluster'] = model.fit_predict(rfm_norm)
    return rfm_norm

def to_excel(df):
    """Mengonversi DataFrame ke format Excel untuk diunduh."""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Segmentasi')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def gabung_data(data_awal, hasil_rfm):
    """Menggabungkan data awal (info pelanggan) dengan hasil segmentasi RFM."""
    kolom_info = [c for c in data_awal.columns if c not in ['Tanggal', 'Total']]
    info_pelanggan = data_awal[kolom_info].drop_duplicates(subset='ID_Pelanggan').set_index('ID_Pelanggan')
    
    hasil_lengkap = hasil_rfm.join(info_pelanggan)
    
    kolom_info_urut = ['ID_Pelanggan', 'Nama'] + [c for c in info_pelanggan.columns if c not in ['ID_Pelanggan','Nama']]
    kolom_rfm_urut = ['Recency', 'Frequency', 'Monetary', 'Cluster', 'Label']
    
    urutan_kolom = kolom_info_urut + kolom_rfm_urut
    urutan_kolom_final = [col for col in urutan_kolom if col in hasil_lengkap.reset_index().columns]
    
    return hasil_lengkap.reset_index().reindex(columns=urutan_kolom_final)


# ------------------------------ Sidebar ------------------------------
st.sidebar.subheader("📂 Upload File")
uploaded_file = st.sidebar.file_uploader("Unggah file Excel atau CSV", type=['csv', 'xlsx'])

# ------------------------------ Halaman Utama ------------------------------
st.title("📊 Analisis Segmentasi Pelanggan")

_, main_col, _ = st.columns([0.05, 0.9, 0.05])
with main_col:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file, sheet_name=0)

            kolom_map = deteksi_kolom(data)
            if kolom_map is None:
                st.error("❌ Tidak bisa menemukan kolom wajib: Tanggal, ID_Pelanggan, Total, Nama.")
            else:
                data = data.rename(columns={v: k for k, v in kolom_map.items()})

                if 'No Telpon' in data.columns:
                    data['No Telpon'] = data['No Telpon'].astype(str)
                    data['No Telpon'] = data['No Telpon'].apply(
                        lambda x: '0' + x if x.startswith('8') else x
                    )

                with st.expander("📄 Data Awal", expanded=False):
                    st.dataframe(data, height=300, hide_index=True, use_container_width=True) 

                rfm = hitung_rfm(data)
                rfm_norm = normalisasi_rfm(rfm[['Recency', 'Frequency', 'Monetary']])
                hasil_cluster = lakukan_clustering(rfm_norm, n_clusters=3)
                rfm['Cluster'] = hasil_cluster['Cluster']

                cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
                sorted_clusters = cluster_summary.sort_values(['Recency', 'Frequency', 'Monetary'], ascending=[True, False, False]).index.tolist()
                label_map = {sorted_clusters[0]: "Loyal", sorted_clusters[1]: "Potensial", sorted_clusters[2]: "Tidak Aktif"}
                rfm['Label'] = rfm['Cluster'].map(label_map)
                cluster_colors = {"Loyal": 'skyblue', "Tidak Aktif": 'pink', "Potensial": 'yellow'}
                
                hasil_lengkap_download = gabung_data(data, rfm)

                with st.expander("📋 Hasil Segmentasi Pelanggan", expanded=True):
                    hasil_tabel_tampil = hasil_lengkap_download[['ID_Pelanggan', 'Nama', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Label']]
                    st.dataframe(hasil_tabel_tampil, height=400, hide_index=True, use_container_width=True)

                with st.expander("📊 Rata-rata RFM per Cluster", expanded=True):
                    cluster_summary['Label'] = cluster_summary.index.map(label_map)
                    st.dataframe(cluster_summary, hide_index=True, use_container_width=True)

                # --- PERUBAHAN 2: Menambah jarak vertikal sebelum judul berikutnya ---
                st.write("")
                st.write("")

                st.subheader("📈 Visualisasi Cluster")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4 style='text-align: center;'>Jumlah Pelanggan per Klaster</h4>", unsafe_allow_html=True)
                    cluster_counts = rfm['Label'].value_counts()
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 5)) 
                    ax_bar.bar(cluster_counts.index, cluster_counts.values, color=[cluster_colors[i] for i in cluster_counts.index])
                    st.pyplot(fig_bar)

                with col2:
                    st.markdown("<h4 style='text-align: center;'>Distribusi Pelanggan</h4>", unsafe_allow_html=True)
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 5)) 
                    ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
                                startangle=90, colors=[cluster_colors[i] for i in cluster_counts.index],
                                textprops={'fontsize': 12}) 
                    ax_pie.axis('equal')
                    st.pyplot(fig_pie)
                
                st.subheader("Scatter Plot Recency vs Monetary")
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                for lbl in cluster_counts.index:
                    cluster_data = rfm[rfm['Label'] == lbl]
                    ax_scatter.scatter(cluster_data['Recency'], cluster_data['Monetary'],
                                        s=80, color=cluster_colors[lbl], label=lbl, alpha=0.7)
                ax_scatter.set_xlabel("Recency (Hari)", fontsize=14)
                ax_scatter.set_ylabel("Monetary (Nilai Transaksi)", fontsize=14)
                ax_scatter.legend(fontsize=12)
                ax_scatter.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_scatter)

                st.subheader("🧩 Ringkasan Tiap Cluster")

                strategi_layanan = {
                    "Tidak Aktif": ["Layanan express dengan harga promo untuk memancing transaksi ulang", "Memberikan voucher diskon ringan untuk transaksi berikutnya", "Memastikan kualitas laundry baik"],
                    "Loyal": ["Dapat kartu poin dengan minimal jumlah cucian", "Tanda khusus pada order untuk jaminan kualitas", "Layanan antar-jemput gratis untuk jarak dekat"],
                    "Potensial": ["Menawarkan paket bundle cuci + setrika dengan harga hemat", "Layanan antar-jemput gratis untuk jumlah cucian tertentu", "Bonus layanan tambahan untuk transaksi pertama di bulan berjalan"]
                }
                strategi_pemasaran = {
                    "Tidak Aktif": ["Mengirim pesan personal via WhatsApp/SMS berisi promo/penawaran khusus"],
                    "Loyal": ["Kirim pesan hangat setelah laundry selesai", "Sistem poin: 1 transaksi = 1 poin, 10 poin = gratis cuci 1x atau setrika 3 kg", "Promo di hari ulang tahun", "Paket langganan bulanan/keluarga dengan harga hemat"],
                    "Potensial": ["Penawaran khusus untuk transaksi pertama bulan berjalan", "Menampilkan testimoni pelanggan untuk meningkatkan kepercayaan"]
                }

                table_data = []
                for cl in sorted_clusters:
                    lbl = label_map[cl]
                    karakteristik_html = (
                        f"<b>Recency:</b> {cluster_summary.loc[cl, 'Recency']} hari<br>"
                        f"<b>Frequency:</b> {cluster_summary.loc[cl, 'Frequency']} kali<br>"
                        f"<b>Monetary:</b> Rp {cluster_summary.loc[cl, 'Monetary']:,.2f}"
                    )
                    layanan_html = "<ul>" + "".join([f"<li>{s}</li>" for s in strategi_layanan[lbl]]) + "</ul>"
                    pemasaran_html = "<ul>" + "".join([f"<li>{s}</li>" for s in strategi_pemasaran[lbl]]) + "</ul>"
                    
                    table_data.append({
                        "Cluster": f"<b>{lbl}</b><br>(Cluster {cl})", "Karakteristik": karakteristik_html,
                        "Strategi Layanan": layanan_html, "Strategi Pemasaran": pemasaran_html
                    })

                df_strategi = pd.DataFrame(table_data)
                html_table = df_strategi.to_html(escape=False, index=False, classes='strategy-table', justify='left')
                st.markdown(html_table, unsafe_allow_html=True)
                
                st.write("---") 
                file_excel = to_excel(hasil_lengkap_download)
                
                st.download_button(
                    label="⬇️ Download Hasil Analisis",
                    data=file_excel,
                    file_name="hasil_segmentasi_pelanggan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
            st.error("Pastikan file Anda memiliki kolom 'Tanggal', 'ID_Pelanggan', 'Total', dan 'Nama'.")
    else:
        st.info("👋 Selamat Datang! Silakan unggah file transaksi pelanggan untuk memulai analisis.")
