import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from io import BytesIO

# Layout halaman tetap "wide"
st.set_page_config(page_title="Segmentasi Pelanggan", layout="wide")

# ------------------------------ CSS Kustom ------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px !important; }
h1 { font-size: 40px !important; font-weight: bold !important; margin-top: -35px !important; margin-bottom: 30px !important; }
h2 { font-size: 28px !important; font-weight: bold !important; }
h3 { font-size: 24px !important; font-weight: bold !important; }
div[data-testid="stExpander"] summary p { font-size: 24px !important; font-weight: bold !important; }
.stDataFrame div, .stDataFrame table { font-size: 22px !important; }
.stDataFrame th, .stDataFrame td { text-align: center !important; padding: 18px !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Fungsi ------------------------------
def deteksi_kolom(data):
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
        elif 'alamat' in c_low or 'address' in c_low:
            kolom['Alamat'] = c
        elif 'telp' in c_low or 'phone' in c_low or 'nomor' in c_low:
            kolom['No Telpon'] = c
    if not required_keys.issubset(kolom.keys()):
        return None
    return kolom

def hitung_rfm(data):
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
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(rfm), columns=rfm.columns, index=rfm.index)

def lakukan_clustering(rfm_norm, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_norm['Cluster'] = model.fit_predict(rfm_norm)
    return rfm_norm

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Segmentasi')
    return output.getvalue()

def gabung_data(data_awal, hasil_rfm):
    kolom_info = [c for c in ['ID_Pelanggan','Nama','Alamat','No Telpon'] if c in data_awal.columns]
    info_pelanggan = data_awal[kolom_info].drop_duplicates(subset='ID_Pelanggan').set_index('ID_Pelanggan')
    hasil_lengkap = hasil_rfm.join(info_pelanggan)
    kolom_info_urut = ['ID_Pelanggan','Nama'] + [c for c in ['Alamat','No Telpon'] if c in info_pelanggan.columns]
    kolom_rfm_urut = ['Recency','Frequency','Monetary','Label']
    urutan_kolom_final = kolom_info_urut + kolom_rfm_urut
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
                data = data.rename(columns={v:k for k,v in kolom_map.items()})
                
                # Format No Telpon
                if 'No Telpon' in data.columns:
                    data['No Telpon'] = data['No Telpon'].astype(str)
                    data['No Telpon'] = data['No Telpon'].apply(lambda x: '0'+x if x.startswith('8') else x)

                with st.expander("📄 Data Awal", expanded=False):
                    st.dataframe(data, height=300, hide_index=True, use_container_width=True)

                # Hitung RFM & Clustering
                rfm = hitung_rfm(data)
                rfm_norm = normalisasi_rfm(rfm[['Recency','Frequency','Monetary']])
                hasil_cluster = lakukan_clustering(rfm_norm, n_clusters=3)
                rfm['Cluster'] = hasil_cluster['Cluster']

                sorted_clusters = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().sort_values(
                    ['Recency','Frequency','Monetary'], ascending=[True,False,False]
                ).index.tolist()
                label_map = {sorted_clusters[0]:"Loyal", sorted_clusters[1]:"Potensial", sorted_clusters[2]:"Tidak Aktif"}
                rfm['Label'] = rfm['Cluster'].map(label_map)
                cluster_colors = {"Loyal": 'skyblue', "Potensial": 'yellow', "Tidak Aktif": 'pink'}

                hasil_lengkap_download = gabung_data(data, rfm)

                # ---------------- Filter ----------------
                with st.expander("📋 Hasil Segmentasi Pelanggan", expanded=True):
                    st.subheader("🔎 Cari Data")
                    keyword = st.text_input("Cari berdasarkan Nama, Alamat, No Telpon atau Label:")

                    kolom_tampil = [c for c in ['ID_Pelanggan','Nama','Alamat','No Telpon','Recency','Frequency','Monetary','Label'] if c in hasil_lengkap_download.columns]
                    hasil_tabel_tampil = hasil_lengkap_download[kolom_tampil]

                    if keyword:
                        hasil_tabel_tampil = hasil_tabel_tampil[
                            hasil_tabel_tampil.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
                        ]

                    st.dataframe(hasil_tabel_tampil, height=400, hide_index=True, use_container_width=True)

                # ---------------- Visualisasi ----------------
                st.subheader("📈 Visualisasi Cluster")
                col1, col2 = st.columns(2)
                cluster_counts = rfm['Label'].value_counts()
                with col1:
                    fig_bar, ax_bar = plt.subplots(figsize=(6,5))
                    ax_bar.bar(cluster_counts.index, cluster_counts.values, color=[cluster_colors[i] for i in cluster_counts.index])
                    st.pyplot(fig_bar)
                with col2:
                    fig_pie, ax_pie = plt.subplots(figsize=(5,5))
                    ax_pie.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
                               startangle=90, colors=[cluster_colors[i] for i in cluster_counts.index],
                               textprops={'fontsize':12})
                    ax_pie.axis('equal')
                    st.pyplot(fig_pie)

                st.subheader("Scatter Plot Recency vs Monetary")
                fig_scatter, ax_scatter = plt.subplots(figsize=(10,6))
                for lbl in cluster_counts.index:
                    cluster_data = rfm[rfm['Label']==lbl]
                    ax_scatter.scatter(cluster_data['Recency'], cluster_data['Monetary'], s=80, color=cluster_colors[lbl], label=lbl, alpha=0.7)
                ax_scatter.set_xlabel("Recency (Hari)", fontsize=14)
                ax_scatter.set_ylabel("Monetary (Nilai Transaksi)", fontsize=14)
                ax_scatter.legend(fontsize=12)
                ax_scatter.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_scatter)

                # ---------------- Ringkasan Cluster ----------------
                st.subheader("📋 Ringkasan Cluster")

                cluster_summary = rfm.groupby('Label')[['Recency','Frequency','Monetary']].mean().reset_index()
                cluster_counts = rfm['Label'].value_counts().to_dict()

                interpretasi_map = {
                    "Loyal": f"Cluster ini terdiri dari {cluster_counts.get('Loyal',0)} pelanggan yang tergolong aktif, dengan transaksi dalam waktu dekat, frekuensi tinggi, dan nilai transaksi besar.",
                    "Tidak Aktif": f"Cluster ini terdiri dari {cluster_counts.get('Tidak Aktif',0)} pelanggan yang tergolong tidak aktif karena sudah lama tidak melakukan transaksi, dengan frekuensi sangat rendah dan nilai belanja kecil.",
                    "Potensial": f"Cluster ini terdiri dari {cluster_counts.get('Potensial',0)} pelanggan yang masih memiliki aktivitas transaksi meskipun tidak rutin, dengan nilai transaksi menengah sehingga digolongkan sebagai pelanggan potensial."
                }

                cluster_summary['Karakteristik'] = cluster_summary.apply(
                    lambda row: 
                        f"Recency : {int(round(row['Recency']))} hari<br>"
                        f"Frequency : {int(round(row['Frequency']))} kali<br>"
                        f"Monetary : Rp {int(round(row['Monetary'])):,.0f}".replace(",", "."),
                    axis=1
                )
                cluster_summary['Interpretasi'] = cluster_summary['Label'].map(interpretasi_map)

                tabel_karakteristik = cluster_summary[['Label','Karakteristik','Interpretasi']].rename(columns={'Label':'Cluster'})

                # CSS untuk tabel
                st.markdown("""
                <style>
                .character-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 18px;
                    text-align: left;
                    box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
                }
                .character-table th, .character-table td {
                    border: 1.5px solid #ccc;
                    padding: 14px;
                    font-size: 18px;
                }
                .character-table th {
                    background-color: #444;
                    color: white;
                    text-align: center;
                    font-size: 20px;
                }
                .character-table td:nth-child(1) { text-align: center; font-weight: bold; }
                .character-table td:nth-child(2) { text-align: left; white-space: pre-line; }
                .character-table td:nth-child(3) { text-align: left; }
                .character-table tr:nth-of-type(even) { background-color: #f9f9f9; }
                .character-table tr:hover { background-color: #f1f1f1; }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(
                    tabel_karakteristik.to_html(index=False, classes="character-table", escape=False),
                    unsafe_allow_html=True
                )

                # ---------------- Download ----------------
                st.write("---")
                file_excel = to_excel(hasil_lengkap_download.drop(columns=['Cluster'], errors='ignore'))
                st.download_button(
                    label="⬇️ Download Hasil Analisis",
                    data=file_excel,
                    file_name="hasil_segmentasi_pelanggan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
            st.error("Pastikan file memiliki kolom 'Tanggal', 'ID_Pelanggan', 'Total', 'Nama', dan opsional 'Alamat'/'No Telpon'.")
    else:
        st.info("👋 Silakan unggah file transaksi pelanggan untuk memulai analisis.")
