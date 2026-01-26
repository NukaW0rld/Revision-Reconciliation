from pydantic import BaseModel
from typing import List, Optional, Dict

class Evidence(BaseModel):
    page: int
    bbox: List[float]
    image_path: Optional[str]

class DeltaItem(BaseModel):
    char_no: Optional[int]
    status: str
    confidence: float
    reasons: List[str]
    scores: Dict[str, float]
    revA: Optional[Evidence]
    revB: Optional[Evidence]

class DeltaPacket(BaseModel):
    run_id: str
    inputs: Dict[str, str]
    items: List[DeltaItem]
