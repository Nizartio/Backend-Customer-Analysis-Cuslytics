import os
import pandas as pd

class DatasetManager:
    """Singleton untuk memuat dataset sekali aja ke memori server."""
    _raw_df: pd.DataFrame = None
    _clean_df: pd.DataFrame = None
    _rfm_df: pd.DataFrame = None
    
    @classmethod
    def load_dataset(cls, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} tidak ditemukan.")
        print(f"Memuat dataset dari {csv_path}...")
        
        # Baca CSV mentah
        cls._raw_df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        
        # Pembersihan dasar
        # Buang Customer ID yang kosong untuk RFM
        valid_customers = cls._raw_df.dropna(subset=['Customer ID']).copy()
        # Buang customer ID yang kosong untuk RFM
        valid_tx = valid_customers[
          (~valid_customers['Invoice'].astype(str).str.startswith('C')) &
          (valid_customers['Quantity'] > 0) &
          (valid_customers['Price'] > 0)
        ].copy()
        
        # Hitung Total Nilai Transaksi (FR-005: Quantity * UnitPrice)
        valid_tx['TotalAmount'] = valid_tx['Quantity'] * valid_tx['Price']
        valid_tx['InvoiceDate'] = pd.to_datetime(valid_tx['InvoiceDate'])
            
        cls._clean_df = valid_tx
        
        observation_date = valid_tx['InvoiceDate'].max() + pd.Timedelta(days=1)                                                                                                                  
                                                                                                                                                                                                     
        rfm = valid_tx.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (observation_date - x.max()).days, # Recency
            'Invoice': 'nunique',                                      # Frequency
            'TotalAmount': 'sum'                                       # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        cls._rfm_df = rfm
        print(f"Dataset berhasil dimuat! {len(rfm)} pelanggan teridentifikasi.")

    @classmethod
    def get_rfm_data(cls) -> pd.DataFrame:
        return cls._rfm_df

    @classmethod
    def get_clean_data(cls) -> pd.DataFrame:
        return cls._clean_df

    @classmethod
    def get_raw_data(cls) -> pd.DataFrame:
        return cls._raw_df