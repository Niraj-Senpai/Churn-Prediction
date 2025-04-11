
# Churn Prediction

A machine learning project to predict customer churn using a supervised learning model. It includes data preprocessing, model training, evaluation, and a Flask-based web application for real-time predictions.

## 📂 Project Structure

```
Churn-Prediction/
├── app.py                         # Flask web app for prediction
├── Customer_Churn_Analysis.ipynb # Jupyter Notebook for EDA and model training
├── gb_model.pkl                  # Pre-trained Gradient Boosting model
├── Telco.csv                     # Customer dataset
├── templates/
│   └── index.html                # HTML form for user input
├── static/
│   └── style.css                 # CSS styles (if any)
└── README.md                     # Project documentation
```

## 📊 Dataset

- **File**: `Telco.csv`
- **Description**: Contains customer demographics, account information, and whether they have churned.
- **Source**: Typically sourced from IBM Sample Data or similar Telco datasets.

## 📘 Notebooks

### `Customer_Churn_Analysis.ipynb`

- Loads and preprocesses the data
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Model building using Gradient Boosting Classifier
- Model evaluation using accuracy, confusion matrix, and classification report

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Niraj-Senpai/Churn-Prediction.git
cd Churn-Prediction
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn flask
```

### 3. Run the Web App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to use the prediction form.

## 🧠 Model Used

- **Gradient Boosting Classifier** from `scikit-learn`
- Trained on encoded and scaled features from the Telco customer dataset
- Saved using `pickle` as `gb_model.pkl`

## 🖥️ Web Interface

A simple Flask app is created to collect user inputs and predict whether a customer is likely to churn. The form collects data such as tenure, monthly charges, contract type, etc., and displays the result on submission.

## 🛠️ Tools & Technologies

- Python 3
- Jupyter Notebook
- Scikit-learn
- Pandas, NumPy
- Flask
- HTML/CSS (basic)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute or raise issues to improve this project!
