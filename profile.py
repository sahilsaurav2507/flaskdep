import json
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Sample Scheme Data
schemes = [
    {
        "scheme_name": "Growth Plan",
        "scheme_type": "Investment",
        "target_gender": "Any",
        "target_age_group": [30, 40],
        "min_investment": 10000,
        "max_investment": 100000,
        "roi": 8.5,
        "risk_level": "Moderate",
        "target_occupation": ["Engineer", "Doctor"],
        "target_income_level": [50000, 100000],
        "target_education_level": "Masters",
        "tax_benefit": "Yes"
    },
    {
        "scheme_name": "Retirement Plan",
        "scheme_type": "Pension",
        "target_gender": "Any",
        "target_age_group": [50, 65],
        "min_investment": 5000,
        "max_investment": 50000,
        "roi": 6.5,
        "risk_level": "Low",
        "target_occupation": ["Any"],
        "target_income_level": [30000, 80000],
        "target_education_level": "Any",
        "tax_benefit": "Yes"
    }
]

# Normalize and calculate score
def calculate_cbf_score(user_profile, schemes):
    scores = []

    for scheme in schemes:
        score = 0
        
        # Age matching
        if scheme['target_age_group'][0] <= user_profile['Age'] <= scheme['target_age_group'][1]:
            score += 1

        # Gender matching
        if scheme['target_gender'] == "Any" or scheme['target_gender'] == user_profile['Gender']:
            score += 1

        # Income matching
        if scheme['target_income_level'][0] <= user_profile['Income'] <= scheme['target_income_level'][1]:
            score += 1

        # Occupation matching
        if scheme['target_occupation'][0] == "Any" or user_profile['Occupation'] in scheme['target_occupation']:
            score += 1

        # Education matching
        if scheme['target_education_level'] == "Any" or scheme['target_education_level'] == user_profile['Education']:
            score += 1

        # Normalize score to percentage
        scores.append({
            "scheme_name": scheme['scheme_name'],
            "buying_probability_score": round((score / 5) * 100, 2)
        })

    return scores

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        user_profile = request.get_json()
        
        # Generate Recommendations
        recommendations = calculate_cbf_score(user_profile, schemes)

        # Prepare output
        output = {
            "user_id": user_profile['Id'],
            "recommendations": recommendations
        }

        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
