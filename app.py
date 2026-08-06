from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/toxicity_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    MolecularWeight = float(request.form["MolecularWeight"])
    LogP = float(request.form["LogP"])
    HBD = float(request.form["HBD"])
    HBA = float(request.form["HBA"])
    RotatableBonds = float(request.form["RotatableBonds"])
    TPSA = float(request.form["TPSA"])

    data = pd.DataFrame([[MolecularWeight, LogP, HBD, HBA, RotatableBonds, TPSA]],
                        columns=["MolecularWeight", "LogP", "HBD", "HBA", "RotatableBonds", "TPSA"])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    toxicity_percentage = round(probabilities[1] * 100, 2)
    confidence = round(max(probabilities) * 100, 2)

    if toxicity_percentage >= 70:
        risk = "HIGH 🔴"
    elif toxicity_percentage >= 40:
        risk = "MEDIUM 🟡"
    else:
        risk = "LOW 🟢"

    result = "TOXIC" if prediction == 1 else "NON-TOXIC"

    return render_template(
        "index.html",
        prediction=result,
        probability=toxicity_percentage,
        confidence=confidence,
        risk=risk
    )


if __name__ == "__main__":
    app.run(debug=True)