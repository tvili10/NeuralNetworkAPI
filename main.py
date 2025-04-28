from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Network import MultilayerPerceptron
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from MNIST_data_handler.Datahandler import Datahandler
import socket

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components lazily
model = None
mnist_data_handler = None

def get_model():
    global model, mnist_data_handler
    if model is None:
        mnist_data_handler = Datahandler()
        model = MultilayerPerceptron([784, 32, 32, 10])
        # Load data without augmentation initially
        X_train, Y_train, X_test, Y_test = mnist_data_handler.get_training_and_test_data(augment=False)
        model.train(X_train, Y_train)
        # Clear training data from memory
        del X_train, Y_train
    return model

class UserDrawing(BaseModel):
    pixels: list


class TrainingExample(BaseModel):
    pixels: list
    label: int

@app.get("/")
def read_root():
    return {"Hello": "World"}
 
@app.post("/predict")
def predict(input: UserDrawing):
    
    if len(input.pixels) != 784:
        raise HTTPException(status_code=400, detail="Input must be a list of 784 pixels")
    
    
   
    model = get_model()
    prediction, _ = model.feed_forward(input.pixels)
    predicted_class = np.argmax(prediction[-1])
    
    probability_distribution = {i: float(prob) for i, prob in enumerate(prediction[-1])}
    sorted_probabilities = dict(sorted(probability_distribution.items(), key=lambda item: item[1], reverse=True))
    print(sorted_probabilities)
    return {"prediction": int(predicted_class), "prob-distribution": sorted_probabilities}

@app.post("/addtrainingexample")
def addTrainingExample(input: TrainingExample):
    if len(input.pixels) != 784:
        raise HTTPException(status_code=400, detail="Input must be a list of 784 pixels")

    global model
    mnist_data_handler = Datahandler()
    mnist_data_handler.add_data(input.pixels, input.label)    
    # Reset model to force retraining with new data
    model = None
    return {"status": "success"}

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

if __name__ == "__main__":
    import uvicorn
    port = 8000
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying next port...")
        port += 1
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
