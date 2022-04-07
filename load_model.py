import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'Purchase_Prediction.pkl' in os.listdir('.'):
            # example url: "https://drive.google.com/uc?id=1YcDspGzsToodglZlovj7dLRUkK-0JrPL&export=download"
            url = "https://drive.google.com/uc?id=1YcDspGzsToodglZlovj7dLRUkK-0JrPL&export=download"
            r = requests.get(url, allow_redirects=True)
            open(r"Purchase_Prediction.pkl", 'wb').write(r.content)
            del r
        with open(r"Purchase_Prediction.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf
