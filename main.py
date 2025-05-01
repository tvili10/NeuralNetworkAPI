from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Network import MultilayerPerceptron
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from MNIST_data_handler.Datahandler import Datahandler
import socket
import os

origins = [
    "http://localhost",
    "http://localhost:3000", 
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://28x28digitrecognizer.netlify.app",
    "https://28x28digitrecognizer.netlify.app/"
]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components lazily
_model = MultilayerPerceptron([784, 32, 32, 10])
_mnist_data_handler = Datahandler()

#X_train, Y_train, X_test, Y_test = _mnist_data_handler.get_training_and_test_data(augment=False)

print("Training model...")
#_model.train(X_train, Y_train)
print("Model trained successfully")

class UserDrawing(BaseModel):
    pixels: list


class TrainingExample(BaseModel):
    pixels: list
    label: int

@app.get("/")
def read_root():
    try:
        return {"Hello": "World"}
    except Exception as e:
        print(f"Root route error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(input: UserDrawing):
    try:
        if len(input.pixels) != 784:
            raise HTTPException(status_code=400, detail="Input must be a list of 784 pixels")
        
        print("Getting model...")
        print("Model loaded successfully")
        
        print("Running prediction...")
        prediction, _ = _model.feed_forward(input.pixels)
        predicted_class = np.argmax(prediction[-1])
        
        probability_distribution = {i: float(prob) for i, prob in enumerate(prediction[-1])}
        sorted_probabilities = dict(sorted(probability_distribution.items(), key=lambda item: item[1], reverse=True))
        print(f"Prediction complete. Class: {predicted_class}, Probabilities: {sorted_probabilities}")
        
        return {"prediction": int(predicted_class), "prob-distribution": sorted_probabilities}
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/addtrainingexample")
def addTrainingExample(input: TrainingExample):
    if len(input.pixels) != 784:
        raise HTTPException(status_code=400, detail="Input must be a list of 784 pixels")

    _mnist_data_handler.add_data(input.pixels, input.label)    
    
    return {"status": "success"}

