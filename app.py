# Enhancing the Flask app to include all queries and generating a beautiful interface for the results page.

# Flask App Initialization with updates
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
import time
import os
# Load model and scaler


current_dir = os.getcwd()  # Gets the current working directory
model_path = os.path.join(current_dir, 'models', 'heart_model_improved.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Initialize Neo4j Handler
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

neo4j_handler =Neo4jHandler("neo4j+s://d6d2eb69.databases.neo4j.io", "neo4j", "GgUHwsQN3_gTDb9CTY8VPebUr1jUsVQj6QGi9tlhLwU")

# Flask App Initialization
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        data = request.form
        age = int(data['age'])
        gender = 1 if data['gender'] == "Male" else 0
        smoking = 1 if data['smoking'] == "Yes" else 0
        family_history = 1 if data['family_history'] == "Yes" else 0
        diabetes = 1 if data['diabetes'] == "Yes" else 0
        obesity = 1 if data['obesity'] == "Yes" else 0
        diet = {"Poor": 0, "Average": 1, "Healthy": 2}[data['diet']]
        exercise = {"None": 0, "Moderate": 1, "Regular": 2}[data['exercise']]
        stress_level = {"Low": 0, "Moderate": 1, "High": 2}[data['stress_level']]
        cholesterol = float(data['cholesterol'])
        resting_bp = float(data['resting_bp'])
        ecg_results = {"Normal": 0, "Abnormal": 1, "Severe": 2}[data['ecg_results']]
        max_heart_rate = float(data['max_heart_rate'])
        chest_pain_type = {"Typical": 0, "Atypical": 1, "Non-Anginal": 2}[data['chest_pain_type']]
        shortness_of_breath = 1 if data['shortness_of_breath'] == "Yes" else 0
        palpitations = 1 if data['palpitations'] == "Yes" else 0

        # Prepare the input for the model
        input_data = np.array([[age, gender, smoking, family_history, diabetes, obesity, diet, exercise,
                                stress_level, cholesterol, resting_bp, ecg_results, max_heart_rate,
                                chest_pain_type, shortness_of_breath, palpitations]])

        # Normalize numerical features
        numerical_features = ["Age", "Cholesterol", "RestingBP", "MaxHeartRate"]
        input_data_df = pd.DataFrame(input_data, columns=[
            "Age", "Gender", "Smoking", "FamilyHistory", "Diabetes", "Obesity", "Diet", "Exercise",
            "StressLevel", "Cholesterol", "RestingBP", "ECGResults", "MaxHeartRate",
            "ChestPainType", "ShortnessOfBreath", "Palpitations"
        ])
        input_data_df[numerical_features] = scaler.transform(input_data_df[numerical_features])

        # Make predictions
        prediction = model.predict(input_data_df)
        probability = model.predict_proba(input_data_df)[0][1]
        time.sleep(3)
        # Fetch additional information from Neo4j
        risk_factors = []
        if smoking: risk_factors.append("Smoking")
        if cholesterol > 240: risk_factors.append("High Cholesterol")
        if resting_bp > 140: risk_factors.append("High Blood Pressure")
        if diabetes: risk_factors.append("Diabetes")
        if obesity: risk_factors.append("Obesity")

        risk_factor_query = """
        MATCH (r:RiskFactor)-[:INCREASES_RISK_OF]->(d:Disease)
        WHERE r.name IN $risk_factors
        RETURN d.name AS Disease, r.name AS RiskFactor
        """
        diseases = neo4j_handler.query(risk_factor_query, {"risk_factors": risk_factors})

        disease_names = [disease["Disease"] for disease in diseases]
        symptom_query = """
        MATCH (d:Disease)-[:PRESENTS_WITH]->(s:Symptom)
        WHERE d.name IN $disease_names
        RETURN s.name AS Symptom
        """
        symptoms = neo4j_handler.query(symptom_query, {"disease_names": disease_names})

        treatment_query = """
        MATCH (d:Disease)-[:TREATED_BY]->(t:Treatment)
        WHERE d.name IN $disease_names
        RETURN d.name AS Disease, t.name AS Treatment
        """
        treatments = neo4j_handler.query(treatment_query, {"disease_names": disease_names})

        preventive_query = """
        MATCH (lm:LifestyleModification)-[:REDUCES]->(r:RiskFactor)
        WHERE r.name IN $risk_factors
        RETURN lm.name AS Modification, r.name AS RiskFactor
        """
        preventive_measures = neo4j_handler.query(preventive_query, {"risk_factors": risk_factors})

        diagnostic_query = """
        MATCH (d:Disease)-[:DIAGNOSED_BY]->(t:DiagnosticTest)
        WHERE d.name IN $disease_names
        RETURN t.name AS Test
        """
        diagnostic_tests = neo4j_handler.query(diagnostic_query, {"disease_names": disease_names})
        
        def remove_duplicates(original_list, key):
            seen = set()  # To track unique values
            unique_list = []
            
            for item in original_list:
                value = item[key]
                if value not in seen:
                    seen.add(value)
                    unique_list.append(item)
            
            return unique_list

        # Remove duplicates from symptoms and diagnostic tests
        symptoms = remove_duplicates(symptoms, 'Symptom')
        diagnostic_tests = remove_duplicates(diagnostic_tests, 'Test')
        
        return render_template(
            "result.html",
            probability=probability,
            diseases=diseases,
            symptoms=symptoms,
            treatments=treatments,
            preventive_measures=preventive_measures,
            diagnostic_tests=diagnostic_tests,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5002)
