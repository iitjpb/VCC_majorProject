from joblib import load
from datetime import datetime
from google.cloud import compute_v1

model = load('cpu_model.joblib')

def predict_and_scale(request):
    today = datetime.now().toordinal()
    prediction = model.predict([[today]])[0]

    client = compute_v1.InstanceGroupManagersClient()
    size = 5 if prediction > 0.7 else 1 if prediction < 0.3 else 3

    client.resize_unary(
        project='sylvan-faculty-452803-v7',
        zone='us-central1-c',
        instance_group_manager='instance-group-1',
        size=size
    )

    return f"Predicted CPU: {prediction}, scaled to {size} instances"
