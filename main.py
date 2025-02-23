from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model (You will later upload this)
model_path = "model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except:
    model = None  # Model not found yet

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
