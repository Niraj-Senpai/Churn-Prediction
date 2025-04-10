import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("gb_model.pkl", "rb"))

# Load training data to get proper column structure
train_df = pd.read_csv("Telco.csv")
train_df.drop(columns=['customerID', 'Churn'], inplace=True, errors='ignore')

# Create tenure_group exactly as during training
train_df['tenure_group'] = pd.cut(train_df['tenure'],
                                  bins=range(1, 80, 12),
                                  right=False,
                                  labels=["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"])
train_df.drop(columns=['tenure'], inplace=True)

# Categorical columns for one-hot encoding
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group'
]

# Get the model's expected column structure
expected_columns = pd.get_dummies(train_df, columns=categorical_cols).columns

@app.route("/")
def loadPage():
    return render_template("home.html", query="")

@app.route("/", methods=["POST"])
def predict():
    try:
        inputs = [request.form.get(f"query{i}", "").strip() for i in range(1, 20)]

        # Create input dataframe
        input_df = pd.DataFrame([inputs], columns=[
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
            'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure'
        ])

        # Convert numerics
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].replace('', 0).astype(int)
        input_df['MonthlyCharges'] = input_df['MonthlyCharges'].replace('', 0).astype(float)
        input_df['TotalCharges'] = input_df['TotalCharges'].replace('', 0).astype(float)
        input_df['tenure'] = input_df['tenure'].replace('', 1).astype(int)

        # Create tenure_group like training
        input_df['tenure_group'] = pd.cut(input_df['tenure'],
                                          bins=range(1, 80, 12),
                                          right=False,
                                          labels=["1 - 12", "13 - 24", "25 - 36", "37 - 48", "49 - 60", "61 - 72"])
        input_df.drop(columns=['tenure'], inplace=True)

        # One-hot encode only categorical columns
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

        # Add missing columns
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns
        input_encoded = input_encoded[expected_columns]

        # Predict
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        result = "This customer is likely to churn!" if prediction == 1 else "This customer is likely to stay."
        confidence = f"Confidence: {probability * 100:.2f}%"

        return render_template("home.html",
                               output1=result,
                               output2=confidence,
                               **{f"query{i}": inputs[i-1] for i in range(1, 20)})
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
