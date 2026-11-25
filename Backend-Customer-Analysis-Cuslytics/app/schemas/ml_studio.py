# app/schemas/ml_studio.py                                                                                                                                                                       
from pydantic import BaseModel, Field
from typing import List, Optional

class TrainRequest(BaseModel):
    k: int = Field(ge=2, le=8, default=4, description="Jumlah klaster K (antara 2 sampai 8)")
    randomSeed: int = Field(default=42, description="Random seed untuk reproduktifitas")

class ClusterProfile(BaseModel):
    clusterId: int
    name: str
    customerCount: int
    percentage: float
    recencyMean: float
    frequencyMean: float
    monetaryMean: float

class ActionItem(BaseModel):
    clusterId: int
    category: str
    title: str
    finding: str
    suggestedAction: str
    limitations: str

class TrainResponse(BaseModel):
    experimentId: str 
    k: int
    inertia: float
    silhouetteScore: Optional[float]
    clusters: List[ClusterProfile]
    recommendations: List[ActionItem]