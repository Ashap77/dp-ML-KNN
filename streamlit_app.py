import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #d63031;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #2ed573;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">❤️ Aplikasi Prediksi Risiko Penyakit Jantung</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar untuk upload data dan konfigurasi
st.sidebar.header("⚙️ Konfigurasi")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV", 
    type=['csv'],
    help="Upload file CSV yang berisi data referensi untuk klasifikasi"
)

# Input nilai K untuk KNN
k_value = st.sidebar.slider("Nilai K untuk KNN", min_value=1, max_value=15, value=5, step=2)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Fungsi untuk klasifikasi menggunakan KNN
def classify_with_knn(input_data, reference_data, k_neighbors):
    # Memisahkan fitur dan target dari data referensi
    X_ref = reference_data.iloc[:, :-1].values  # Semua kolom kecuali yang terakhir
    y_ref = reference_data.iloc[:, -1].values   # Kolom terakhir sebagai target
    
    # Membuat dan melatih model KNN
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_ref, y_ref)
    
    # Prediksi untuk data input
    input_array = np.array(input_data).reshape(1, -1)
    prediction = knn.predict(input_array)[0]
    prediction_proba = knn.predict_proba(input_array)[0]
    
    # Mencari tetangga terdekat untuk analisis
    distances, indices = knn.kneighbors(input_array, n_neighbors=k_neighbors)
    nearest_neighbors = reference_data.iloc[indices[0]]
    
    return prediction, prediction_proba, nearest_neighbors, distances[0]

# Main application
if uploaded_file is not None:
    # Load data referensi
    reference_df = load_data(uploaded_file)
    
    # Validasi data (harus berisi 0 dan 1)
    if not all(reference_df.select_dtypes(include=[np.number]).isin([0, 1]).all()):
        st.error("⚠️ Dataset harus berisi nilai 0 dan 1 saja!")
        st.stop()
    
    # Display dataset info
    with st.expander("📊 Informasi Dataset Referensi"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(reference_df))
        with col2:
            st.metric("Jumlah Fitur", len(reference_df.columns)-1)
        with col3:
            high_risk_count = sum(reference_df.iloc[:, -1] == 1)
            st.metric("High Risk Cases", high_risk_count)
        
        st.dataframe(reference_df.head(10))
        
        # Distribusi target
        target_counts = reference_df.iloc[:, -1].value_counts()
        fig_dist = px.pie(values=target_counts.values, 
                         names=['Low Risk (0)', 'High Risk (1)'],
                         title="Distribusi Target dalam Dataset",
                         color_discrete_map={'Low Risk (0)': '#2ed573', 'High Risk (1)': '#ff4757'})
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Input form untuk klasifikasi
    st.markdown("## 🔮 Input Data untuk Klasifikasi")
    
    feature_names = reference_df.columns[:-1]  # Semua kolom kecuali yang terakhir
    target_name = reference_df.columns[-1]     # Kolom terakhir sebagai target
    
    with st.form("classification_form"):
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 📝 Masukkan Nilai 0 atau 1 untuk Setiap Faktor:")
        
        input_data = {}
        
        # Membuat input dalam bentuk grid
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_names):
            with cols[i % num_cols]:
                # Menampilkan nama fitur yang lebih readable
                display_name = feature.replace('_', ' ').replace('.', ' ').title()
                
                input_data[feature] = st.selectbox(
                    f"**{display_name}**",
                    options=[0, 1],
                    format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)",
                    key=f"input_{feature}",
                    help=f"Pilih 0 (Tidak) atau 1 (Ya) untuk {display_name}"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("🔍 Klasifikasi Risiko", type="primary", use_container_width=True)
        
        if submitted:
            # Konversi input ke list
            input_values = [input_data[feature] for feature in feature_names]
            
            # Klasifikasi menggunakan KNN
            try:
                prediction, prediction_proba, neighbors, distances = classify_with_knn(
                    input_values, reference_df, k_value
                )
                
                # Menampilkan hasil
                st.markdown("## 📋 Hasil Klasifikasi")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="risk-high">⚠️ HIGH RISK</div>', unsafe_allow_html=True)
                        confidence = prediction_proba[1] * 100
                        st.error(f"**Prediksi: RISIKO TINGGI**")
                        st.write(f"Confidence: {confidence:.1f}%")
                    else:
                        st.markdown('<div class="risk-low">✅ LOW RISK</div>', unsafe_allow_html=True)
                        confidence = prediction_proba[0] * 100
                        st.success(f"**Prediksi: RISIKO RENDAH**")
                        st.write(f"Confidence: {confidence:.1f}%")
                
                with col2:
                    # Visualisasi probabilitas
                    prob_df = pd.DataFrame({
                        'Risk Level': ['Low Risk (0)', 'High Risk (1)'],
                        'Probability': prediction_proba
                    })
                    
                    fig_prob = px.bar(prob_df, x='Risk Level', y='Probability',
                                    title='Probabilitas Klasifikasi',
                                    color='Risk Level',
                                    color_discrete_map={'Low Risk (0)': '#2ed573', 'High Risk (1)': '#ff4757'})
                    fig_prob.update_layout(showlegend=False, yaxis_range=[0, 1])
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                # Detail analisis
                st.markdown("## 🔍 Detail Analisis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Data yang diinput
                    st.markdown("### 📊 Data Input Anda:")
                    input_display = pd.DataFrame({
                        'Faktor Risiko': [name.replace('_', ' ').title() for name in feature_names],
                        'Nilai': [f"{'Ya' if val == 1 else 'Tidak'} ({val})" for val in input_values]
                    })
                    st.dataframe(input_display, use_container_width=True)
                
                with col2:
                    # Tetangga terdekat
                    st.markdown(f"### 🎯 {k_value} Tetangga Terdekat:")
                    neighbors_display = neighbors.copy()
                    neighbors_display['Jarak'] = distances
                    neighbors_display['Target'] = neighbors_display.iloc[:, -2].map({0: 'Low Risk', 1: 'High Risk'})
                    st.dataframe(neighbors_display[['Target', 'Jarak']].round(3), use_container_width=True)
                
                # Confidence box
                st.markdown(f'''
                <div class="confidence-box">
                    <h4>📊 Informasi Klasifikasi</h4>
                    <p><strong>Algoritma:</strong> K-Nearest Neighbors (KNN)</p>
                    <p><strong>K Value:</strong> {k_value}</p>
                    <p><strong>Total Data Referensi:</strong> {len(reference_df)}</p>
                    <p><strong>Confidence Score:</strong> {max(prediction_proba):.2%}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Voting detail
                with st.expander("🗳️ Detail Voting dari Tetangga Terdekat"):
                    voting_detail = neighbors.iloc[:, -1].value_counts()
                    st.write("**Hasil Voting:**")
                    
                    for risk_level, count in voting_detail.items():
                        risk_name = "High Risk" if risk_level == 1 else "Low Risk"
                        st.write(f"- {risk_name}: {count} votes")
                    
                    st.write(f"\n**Keputusan:** {'High Risk' if prediction == 1 else 'Low Risk'} (Mayoritas voting)")
                
                # Rekomendasi
                st.markdown("## 💡 Rekomendasi")
                if prediction == 1:
                    st.warning("""
                    **⚠️ Rekomendasi untuk Risiko Tinggi:**
                    - 🏥 Konsultasi segera dengan dokter spesialis jantung
                    - 🔬 Lakukan pemeriksaan EKG dan echocardiogram  
                    - 📊 Pantau tekanan darah dan kolesterol secara rutin
                    - 🥗 Ubah pola makan: kurangi garam, lemak jenuh, dan kolesterol
                    - 🏃‍♂️ Olahraga teratur minimal 30 menit/hari
                    - 🚭 Hindari merokok dan alkohol
                    - 😌 Kelola stress dengan baik
                    """)
                else:
                    st.info("""
                    **✅ Rekomendasi untuk Risiko Rendah:**
                    - 🎯 Pertahankan gaya hidup sehat saat ini
                    - 📅 Pemeriksaan kesehatan rutin setahun sekali  
                    - 💪 Lanjutkan olahraga teratur
                    - 🥬 Konsumsi buah dan sayuran yang cukup
                    - ⚖️ Jaga berat badan ideal
                    - 😴 Istirahat yang cukup (7-8 jam/hari)
                    - 🧘‍♀️ Kelola stress dengan baik
                    """)
            
            except Exception as e:
                st.error(f"Error dalam klasifikasi: {str(e)}")

else:
    # Halaman welcome
    st.markdown("""
    ## 🚀 Selamat Datang di Aplikasi Prediksi Penyakit Jantung
    
    Aplikasi ini menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk mengklasifikasi risiko penyakit jantung 
    berdasarkan faktor-faktor risiko biner (0/1) yang Anda masukkan.
    
    ### 📋 Cara Menggunakan:
    1. **Upload Dataset**: Unggah file CSV dengan data referensi di sidebar
    2. **Input Data**: Masukkan nilai **0** (Tidak) atau **1** (Ya) untuk setiap faktor risiko
    3. **Klasifikasi**: Klik tombol klasifikasi untuk mendapatkan hasil
    4. **Lihat Hasil**: Dapatkan prediksi **High Risk** atau **Low Risk**
    
    ### 📊 Format Dataset:
    - File CSV dengan kolom berisi nilai **0** atau **1** saja
    - Setiap baris = data pasien
    - Kolom terakhir = target: **0** (Low Risk) atau **1** (High Risk)
    
    ### 🎯 Fitur Aplikasi:
    - ✅ Input sederhana dengan pilihan 0/1
    - ✅ Klasifikasi real-time menggunakan KNN
    - ✅ Analisis tetangga terdekat
    - ✅ Visualisasi probabilitas dan voting
    - ✅ Rekomendasi berdasarkan hasil
    
    **Mulai dengan mengupload dataset Anda di sidebar!** 👈
    """)
    
    # Sample data format
    with st.expander("📖 Contoh Format Dataset"):
        sample_data = pd.DataFrame({
            'age_over_50': [1, 1, 0, 1, 0],
            'chest_pain': [1, 0, 1, 1, 0], 
            'high_blood_pressure': [1, 1, 0, 1, 0],
            'high_cholesterol': [1, 1, 0, 1, 0],
            'smoking': [1, 1, 0, 1, 0],
            'diabetes': [0, 1, 0, 1, 0],
            'target': [1, 1, 0, 1, 0]  # 0=Low Risk, 1=High Risk
        })
        st.dataframe(sample_data)
        st.caption("✅ Semua nilai harus 0 atau 1. Target: 0 = Low Risk, 1 = High Risk")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #74b9ff;'>
    💙 Aplikasi Klasifikasi Risiko Penyakit Jantung - K-Nearest Neighbors
    <br>
    <small>⚠️ Hasil klasifikasi ini bukan diagnosis medis. Selalu konsultasikan dengan tenaga medis profesional.</small>
</div>
""", unsafe_allow_html=True)


