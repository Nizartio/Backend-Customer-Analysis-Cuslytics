from contextlib import asynccontextmanager                                                                                                                                                       
from fastapi import FastAPI, HTTPException                                                                                                                                                       
from fastapi.middleware.cors import CORSMiddleware                                                                                                                                               
import os                                                                                                                                                                                        
                                                                                                                                                                                                  
from app.services.data_loader import DatasetManager                                                                                                                                              
from app.schemas.ml_studio import TrainRequest, TrainResponse                                                                                                                                    
from app.ml_engine.segmentation import execute_kmeans_pipeline                                                                                                                                   
                                                                                                                                                                                                  
# Path ke file CSV yang ada di folder root proyek                                                                                                                                                
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Year 2009-2010.csv")                                                                                                             
                                                                                                                                                                                                  
@asynccontextmanager                                                                                                                                                                             
async def lifespan(app: FastAPI):                                                                                                                                                                
    # Dijalankan saat server mulai                                                                                                                                                               
    print("Inisialisasi aplikasi CusLytics...")                                                                                                                                                  
    try:                                                                                                                                                                                         
        DatasetManager.load_dataset(CSV_PATH)                                                                                                                                                    
    except Exception as e:                                                                                                                                                                       
        print(f"Peringatan: Gagal memuat CSV saat startup: {e}")                                                                                                                                 
    yield                                                                                                                                                                                        
    # Dijalankan saat server ditutup                                                                                                                                                             
    print("Mematikan server CusLytics...")                                                                                                                                                       
                                                                                                                                                                                                  
app = FastAPI(                                                                                                                                                                                   
    title="CusLytics API",                                                                                                                                                                       
    version="1.0.0",                                                                                                                                                                             
    description="Customer Analytics & Continuous ML Studio API",                                                                                                                                 
    lifespan=lifespan                                                                                                                                                                            
)                                                                                                                                                                                                
                                                                                                                                                                                                  
# Konfigurasi CORS agar frontend React bisa memanggil API                                                                                                                                        
app.add_middleware(                                                                                                                                                                              
    CORSMiddleware,                                                                                                                                                                              
    allow_origins=["*"], # Saat deploy, bisa diisi domain frontend kamu                                                                                                                          
    allow_credentials=True,                                                                                                                                                                      
    allow_methods=["*"],                                                                                                                                                                         
    allow_headers=["*"],                                                                                                                                                                         
)                                                                                                                                                                                                
                                                                                                                                                                                                  
@app.get("/")                                                                                                                                                                                    
def read_root():                                                                                                                                                                                 
    return {"message": "CusLytics API is running"}                                                                                                                                               
                                                                                                                                                                                                  
@app.get("/api/v1/dataset/status")                                                                                                                                                               
def get_dataset_status():                                                                                                                                                                        
    rfm = DatasetManager.get_rfm_data()                                                                                                                                                          
    clean = DatasetManager.get_clean_data()                                                                                                                                                      
    if rfm is None or clean is None:                                                                                                                                                             
        raise HTTPException(status_code=503, detail="Dataset belum siap di server.")                                                                                                             
                                                                                                                                                                                                  
    return {                                                                                                                                                                                     
        "filename": "Year 2009-2010.csv",                                                                                                                                                        
        "status": "ready",                                                                                                                                                                       
        "cleanTransactionsCount": len(clean),                                                                                                                                                    
        "uniqueCustomersCount": len(rfm)                                                                                                                                                         
    }                                                                                                                                                                                            
                                                                                                                                                                                                  
@app.post("/api/v1/ml/train", response_model=TrainResponse)                                                                                                                                      
def train_model(payload: TrainRequest):                                                                                                                                                          
    rfm = DatasetManager.get_rfm_data()                                                                                                                                                          
    if rfm is None:                                                                                                                                                                              
        raise HTTPException(status_code=503, detail="Dataset belum dimuat.")                                                                                                                     
                                                                                                                                                                                                  
    # Jalankan Continuous ML Pipeline                                                                                                                                                            
    result = execute_kmeans_pipeline(rfm, k=payload.k, random_seed=payload.randomSeed)                                                                                                           
    return result