import joblib
import numpy as np
import yaml
import os

# Load model and YAML once at startup
model_path = os.path.join("models", "drowsiness_pipeline.pkl")
config_path = os.path.join("models", "model_config.yml")

loaded_model = joblib.load(model_path)

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

scaler = loaded_model['scaler']
reducer = loaded_model['reducer']
gmm = loaded_model['gmm']
active_cluster = loaded_model['active_cluster']
drowsy_cluster = loaded_model['drowsy_cluster']

def get_drowsy_state(vit_features, traditional_features):
    vit_array = np.array(vit_features)
    trad_array = np.array(traditional_features)

    vit_scaled = scaler.transform(vit_array)
    vit_reduced = reducer.transform(vit_scaled)
    combined = np.hstack((vit_reduced, trad_array))

    labels = gmm.predict(combined)
    states = ['Active' if l == active_cluster else 'Drowsy' for l in labels]

    # For simplicity, return the majority prediction
    return max(set(states), key=states.count)
