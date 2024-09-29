"""
Pydantic schemas for application data validation.
"""

from typing import List

from pydantic import BaseModel


class PredictInput(BaseModel):
    """
    Schema for input data used in predictions.

    Attributes:
        Material_A_Charged_Amount: Amount of material A charged.
        Material_B_Charged_Amount: Amount of material B charged.
        Reactor_Volume: Volume of the reactor.
        Material_A_Final_Concentration_Previous_Batch:
        Final concentration of material A from the previous batch.
    """

    Material_A_Charged_Amount: List[List[float]]
    Material_B_Charged_Amount: List[List[float]]
    Reactor_Volume: List[List[float]]
    Material_A_Final_Concentration_Previous_Batch: List[List[float]]
