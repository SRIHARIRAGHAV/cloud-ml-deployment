from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Placeholder model function
def dummy_model(features):
    return [sum(features[0])]  # Example: Sum of input features as prediction

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = dummy_model(features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
