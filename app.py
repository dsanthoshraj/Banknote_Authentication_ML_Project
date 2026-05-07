from fastapi import FastAPI
from BankNote import BankNote
import pickle

app = FastAPI()
pickle_in = open('bnClassifier.pkl', 'rb')
bnClassifier = pickle.load(pickle_in)

@app.post("/predict_banknote")
def predict_banknote(data: BankNote):
    variance = data.variance
    skewness = data.skewness
    curtosis = data.curtosis
    entropy = data.entropy

    prediction = bnClassifier.predict([[variance, skewness, curtosis, entropy]])

    if prediction[0] > 0.5:
        predict = "The banknote is not authentic"
    else: 
        predict = "The banknote is authentic"

    return {
        "prediction": predict
    }

@app.get("/")
def read_root():
    return {"Welcome to the Bank Note Authentication API! Please use the /predict_banknote endpoint to predict if a bank note is authentic or not."}   

