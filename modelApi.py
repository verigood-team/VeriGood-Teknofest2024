from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Dict
import torch
from mini_pyabsa import AspectSentimentTripletExtraction as ASTE

checkpoint_path = r'checkpoints\dataset_45.31'

model = ASTE.AspectSentimentTripletExtractor(checkpoint_path)

app = FastAPI()

class Item(BaseModel):
    text: str = Field(..., example="""Example text""")

class PredictionResult(BaseModel):
    entity_list: List[str]
    results: List[Dict[str, str]]

@app.post("/predict/", response_model=PredictionResult)
async def predict(item: Item):
    
    input_text = item.text
    
    with torch.no_grad():
        entity_list, results = model.predict(input_text)  

        formatted_result = {
            "entity_list": entity_list,
            "results": [
                {
                    "entity": entity,
                    "sentiment": sentiment
                }
                for result_dict in results
                for entity, sentiment in result_dict.items()
            ]
        }

    return formatted_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
