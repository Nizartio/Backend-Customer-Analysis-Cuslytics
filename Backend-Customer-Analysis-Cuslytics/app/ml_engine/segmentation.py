# app/ml_engine/segmentation.py                                                                                                                                                                  
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import uuid

def execute_kmeans_pipeline(rfm_df: pd.DataFrame, k: int, random_seed: int = 42) -> dict:
    # 1. Ambil fitur RFM & terapkan Log Transform (karena data ritel biasanya skewed)                                                                                                            
    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    features_log = np.log1p(features) # log(1 + x) untuk cegah nilai 0/negatif
    
    # 2. Standarisasi Fitur (FR-010)                                                                                                                                                             
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_log)
    
    # 3. Latih K-Means (FR-011)                                                                                                                                                                  
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # 4. Hitung Metrik Evaluasi (FR-013)                                                                                                                                                         
    inertia = float(kmeans.inertia_)
    try:
        # Silhouette sampling agar komputasi tidak freeze jika data ribuan                                                                                                                       
        sil_score = float(silhouette_score(scaled_data, labels, sample_size=2000, random_state=random_seed))
    except Exception:
        sil_score = None
        
    # 5. Gabungkan hasil klaster ke dataframe asli untuk dihitung karakteristiknya
    df_clustered = rfm_df.copy()
    df_clustered['Cluster'] = labels
    total_customers = len(df_clustered)
    
    # Benchmark rata-rata populasi
    pop_med_r = rfm_df['Recency'].median()
    pop_med_f = rfm_df['Frequency'].median()
    pop_med_m = rfm_df['Monetary'].median()
    
    clusters_output = []
    recommendations_output = []
    
    for c_id in range(k):
        c_data = df_clustered[df_clustered['Cluster'] == c_id]
        count = len(c_data)
        pct = round((count / total_customers) * 100, 2)
        
        r_mean = round(float(c_data['Recency'].mean()), 1)
        f_mean = round(float(c_data['Frequency'].mean()), 1)
        m_mean = round(float(c_data['Monetary'].mean()), 2)
        
        # Rule-Based Naming & Business Action (Aturan SRS Hal 10)                                                                                                                                
        is_recent = r_mean <= pop_med_r
        is_frequent = f_mean >= pop_med_f
        is_high_spender = m_mean >= pop_med_m
        
        if is_recent and is_frequent and is_high_spender:
            name = "Champions / High-Spenders"
            rec_title = "Program Loyalitas & Akses Eksklusif"
            rec_action = "Berikan preview katalog baru lebih awal dan layanan pelanggan prioritas."
        elif is_recent and not is_frequent:
            name = "New / Promising Customers"
            rec_title = "Aktivasi Pembelian Kedua"
            rec_action = "Kirim voucher selamat datang untuk pembelian kedua dalam 14 hari."
        elif not is_recent and (is_frequent or is_high_spender):
            name = "At Risk Customers"
            rec_title = "Kampanye Win-Back Personal"
            rec_action = "Kirim email personalisasi menanyakan kendala atau tawarkan produk rekomendasi."
        else:
            name = "Hibernating / Low Engagement"
            rec_title = "Re-engagement Hemat Biaya"
            rec_action = "Gunakan saluran murah (newsletter standar) dan hindari promosi berbiaya tinggi."
            
        clusters_output.append({
            "clusterId": c_id,
            "name": name,
            "customerCount": count,
            "percentage": pct,
            "recencyMean": r_mean,
            "frequencyMean": f_mean,
            "monetaryMean": m_mean
        })
        
        recommendations_output.append({
            "clusterId": c_id, 
            "category": "Marketing Playbook", 
            "title": rec_title,
            "finding": f"Rata-rata Recency {r_mean} hari, Frekuensi {f_mean}x transaksi, Nilai belanja £{m_mean}.",
            "suggestedAction": rec_action, 
            "limitations": "Dataset tidak memuat data biaya promosi sehingga efektivitas diskon belum teruji." 
        }) 
        
    return { 
        "experimentId": f"exp_{uuid.uuid4().hex[:8]}",
        "k": k,
        "inertia": round(inertia, 2), 
        "silhouetteScore": round(sil_score, 3) if sil_score is not None else None, 
        "clusters": clusters_output, 
        "recommendations": recommendations_output 
    }