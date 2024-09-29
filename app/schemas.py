from pydantic import BaseModel
from typing import List

class PredictInput(BaseModel):
    Material_A_Charged_Amount: List[List[float]]
    Material_B_Charged_Amount: List[List[float]]
    Reactor_Volume: List[List[float]]
    Material_A_Final_Concentration_Previous_Batch: List[List[float]]
