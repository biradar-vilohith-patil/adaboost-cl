import pickle
import pandas as pd
import os
import numpy as np

def load_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, 'models', 'multibagger_adaboost_pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)

def run_inference(input_dict):
    pipeline = load_pipeline()
    df_input = pd.DataFrame([input_dict])
    
    raw_prediction = pipeline.predict(df_input)[0]
    prediction = int(float(raw_prediction))
        
    probabilities = pipeline.predict_proba(df_input)[0]
    confidence = np.max(probabilities) * 100
    
    return prediction, confidence