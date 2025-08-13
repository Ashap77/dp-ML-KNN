import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #e74c3c;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .result-high {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    .result-low {
        background: linear-gradient(90deg, #27ae60, #229954);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
    }
    .input-container {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #3498db;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">❤️ Prediksi Risiko Penyakit Jantung</h1>', unsafe_allow_html=True)

# Sidebar untuk upload CSV
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=['csv'],
    help="Upload dataset dengan kolom binary (0/1) dan target di kolom terakhir"
)

# Parameter KNN
k_value = st.sidebar.slider("🎯 Nilai K untuk KNN", 3, 15, 5, 2)

if uploaded_file is not None:
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validasi data binary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not df[numeric_cols].isin([0, 1]).all().all():
            st.error("⚠️ Dataset harus berisi nilai 0 dan 1 saja!")
            st.stop()
        
        # Info dataset di sidebar
        st.sidebar.success(f"✅ Dataset loaded: {len(df)} rows")
        st.sidebar.info(f"📊 Features: {len(df.columns)-1}")
        
        # Ambil nama kolom (kecuali kolom terakhir yang adalah target)
        feature_columns = df.columns[:-1]
        target_column = df.columns[-1]
        
        # Input Section
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("### 📝 Masukkan Data Pasien")
        st.markdown("*Pilih 0 (Tidak) atau 1 (Ya) untuk setiap faktor risiko*")
        
        # Input form
        with st.form("input_form"):
            input_data = {}
            
            # Membuat 2 kolom untuk layout yang rapi
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(feature_columns):
                # Alternate between columns
                current_col = col1 if i % 2 == 0 else col2
                
                with current_col:
                    # Format nama feature untuk display
                    display_name = feature.replace('_', ' ').replace('-', ' ').title()
                    
                    input_data[feature] = st.selectbox(
                        f"**{display_name}**",
                        options=[0, 1],
                        format_func=lambda x: f"❌ Tidak ({x})" if x == 0 else f"✅ Ya ({x})",
                        key=f"input_{feature}",
                        help=f"Apakah pasien memiliki {display_name.lower()}?"
                    )
            
            # Submit button
            submit_button = st.form_submit_button(
                "🔍 Analisis Risiko",
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Proses klasifikasi ketika form disubmit
        if submit_button:
            # Persiapkan data untuk klasifikasi
            X = df[feature_columns].values  # Features dari dataset
            y = df[target_column].values    # Target dari dataset
            
            # Convert input ke array
            input_array = np.array([list(input_data.values())])
            
            # Inisialisasi dan fit model KNN
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X, y)
            
            # Prediksi
            prediction = knn.predict(input_array)[0]
            prediction_proba = knn.predict_proba(input_array)[0]
            
            # Cari tetangga terdekat
            distances, indices = knn.kneighbors(input_array, n_neighbors=k_value)
            
            # Tampilkan hasil
            st.markdown("---")
            st.markdown("## 📊 Hasil Analisis")
            
            # Hasil prediksi utama
            if prediction == 1:
                st.markdown(
                    '<div class="result-high">⚠️ RISIKO TINGGI<br><small>High Risk untuk Penyakit Jantung</small></div>',
                    unsafe_allow_html=True
                )
                confidence = prediction_proba[1] * 100
                st.error(f"**Tingkat Kepercayaan: {confidence:.1f}%**")
            else:
                st.markdown(
                    '<div class="result-low">✅ RISIKO RENDAH<br><small>Low Risk untuk Penyakit Jantung</small></div>',
                    unsafe_allow_html=True
                )
                confidence = prediction_proba[0] * 100
                st.success(f"**Tingkat Kepercayaan: {confidence:.1f}%**")
            
            # Detail analisis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📋 Data Input")
                input_df = pd.DataFrame({
                    'Faktor Risiko': [col.replace('_', ' ').title() for col in feature_columns],
                    'Nilai': [f"{'Ya' if val == 1 else 'Tidak'}" for val in input_data.values()]
                })
                st.dataframe(input_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Analisis KNN")
                st.metric("K Neighbors", k_value)
                st.metric("Dataset Size", len(df))
                
                # Voting dari tetangga
                neighbor_targets = y[indices[0]]
                high_risk_votes = sum(neighbor_targets == 1)
                low_risk_votes = sum(neighbor_targets == 0)
                
                st.write("**Voting Tetangga:**")
                st.write(f"• High Risk: {high_risk_votes} votes")
                st.write(f"• Low Risk: {low_risk_votes} votes")
            
            # Progress bar untuk confidence
            st.markdown("### 📈 Tingkat Kepercayaan")
            st.progress(max(prediction_proba))
            
            # Rekomendasi
            st.markdown("---")
            st.markdown("## 💡 Rekomendasi")
            
            if prediction == 1:
                st.warning("""
                **⚠️ Untuk Risiko Tinggi:**
                
                🏥 **Segera konsultasi dokter spesialis jantung**
                
                📋 **Pemeriksaan yang disarankan:**
                - EKG (Elektrokardiogram)
                - Echocardiogram  
                - Tes darah lengkap
                - Tes tekanan darah 24 jam
                
                🎯 **Perubahan gaya hidup:**
                - Diet rendah garam dan lemak
                - Olahraga teratur (konsultasi dokter dulu)
                - Berhenti merokok
                - Kelola stress
                - Kontrol berat badan
                """)
            else:
                st.info("""
                **✅ Untuk Risiko Rendah:**
                
                🎉 **Pertahankan kondisi kesehatan saat ini**
                
                📅 **Pemeriksaan rutin:**
                - Check-up kesehatan tahunan
                - Monitor tekanan darah
                - Tes kolesterol berkala
                
                💪 **Gaya hidup sehat:**
                - Olahraga teratur 30 menit/hari
                - Diet seimbang dengan buah & sayur
                - Tidur cukup 7-8 jam
                - Hindari rokok dan alkohol berlebihan
                - Kelola stress dengan baik
                """)
            
            # Disclaimer
            st.markdown("---")
            st.info("⚠️ **Disclaimer:** Hasil ini hanya untuk referensi dan bukan diagnosis medis. Selalu konsultasikan dengan tenaga medis profesional untuk diagnosis dan pengobatan yang tepat.")

else:
    # Welcome screen ketika belum upload file
    st.markdown("""
    ## 🚀 Cara Menggunakan Aplikasi
    
    ### 📤 Langkah 1: Upload Dataset
    - Klik **"Browse files"** di sidebar
    - Upload file CSV dengan format yang benar
    - Dataset harus berisi nilai **0** dan **1** saja
    
    ### 📝 Langkah 2: Input Data
    - Isi setiap faktor risiko dengan **0 (Tidak)** atau **1 (Ya)**
    - Klik tombol **"Analisis Risiko"**
    
    ### 📊 Langkah 3: Lihat Hasil  
    - Dapatkan hasil **High Risk** atau **Low Risk**
    - Lihat tingkat kepercayaan dan rekomendasi
    
    ---
    
    ## 📋 Format Dataset CSV
    
    Dataset harus memiliki struktur seperti ini:
    """)
    
    # Contoh format dataset
    sample_df = pd.DataFrame({
        'age_over_50': [1, 0, 1, 0, 1],
        'chest_pain': [1, 0, 1, 1, 0],
        'high_bp': [1, 0, 1, 0, 1], 
        'cholesterol': [1, 1, 1, 0, 1],
        'smoking': [1, 0, 1, 0, 0],
        'diabetes': [0, 0, 1, 0, 1],
        'target': [1, 0, 1, 0, 1]
    })
    
    st.dataframe(sample_df, hide_index=True, use_container_width=True)
    
    st.markdown("""
    **Keterangan:**
    - Semua kolom kecuali **target** adalah faktor risiko (0 atau 1)
    - **target**: 0 = Low Risk, 1 = High Risk
    - Setiap baris adalah data satu pasien
    
    ---
    
    ## 🎯 Tentang Algoritma KNN
    
    Aplikasi ini menggunakan **K-Nearest Neighbors** untuk klasifikasi:
    - Mencari tetangga terdekat dari data input Anda
    - Melakukan voting berdasarkan K tetangga terdekat  
    - Memberikan hasil klasifikasi High Risk/Low Risk
    - Menampilkan tingkat kepercayaan prediksi
    
    **Mulai dengan upload dataset Anda di sidebar!** 👈
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; padding: 20px;">'
    '💙 Aplikasi Prediksi Risiko Penyakit Jantung | Powered by K-Nearest Neighbors'
    '</div>',
    unsafe_allow_html=True
)
