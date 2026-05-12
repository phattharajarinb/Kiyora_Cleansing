from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from kiyora_model import predict_customer, get_dashboard_summary, get_prediction_history

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/index.html")
def index_page():
    return send_from_directory(".", "index.html")


@app.route("/dashboard.html")
def dashboard_page():
    return send_from_directory(".", "dashboard.html")


@app.route("/products.html")
def products_page():
    return send_from_directory(".", "products.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_customer(data)
    return jsonify(result)


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    return jsonify(get_dashboard_summary())


@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(get_prediction_history())

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=False)