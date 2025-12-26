from typing import List

from typing import List
from pydantic import BaseModel, Field
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI

class EngineData(BaseModel):
    unit_nr: int
    time_cycles: int
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class InferencePayload(BaseModel):
    engine_data_sequence: List[EngineData] = Field(min_length=1, max_length=50)

    def to_dataframe(self) -> pd.DataFrame:
        data_dicts = [edata.model_dump() for edata in self.engine_data_sequence]
        return pd.DataFrame(data_dicts)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup logic goes here
    print("Loading model...")
    
    yield  # This separates startup from shutdown
    
    # 2. Shutdown logic goes here
    print("Shutting down...")

# We pass the lifespan function to the FastAPI app
app = FastAPI(lifespan=lifespan)