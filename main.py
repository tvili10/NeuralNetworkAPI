from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Network import MultilayerPerceptron
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from MNIST_data_handler.Datahandler import Datahandler


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




model = MultilayerPerceptron([784, 32, 32, 10])
mnist_data_handler = Datahandler()
X_train, Y_train, X_test, Y_test = mnist_data_handler.get_training_and_test_data()
model.train(X_train, Y_train)
model.test(X_test, Y_test)

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

    
    mnist_data_handler.add_data(input.pixels, input.label)    
    return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)