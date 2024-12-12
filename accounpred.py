import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample Data: Region Features with historical growth rates
data = {
    
    "Kishan Vikas Patra": {
        "region_name": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli", "Vellore", "Erode", "Dindigul", "Kancheepuram"],
        "acc_open_2019": [165, 381, 248, 240, 263, 142, 166, 358, 347, 297],
        "acc_open_2020": [246, 235, 286, 323, 440, 336, 169, 357, 286, 329],
        "acc_open_2021": [346, 321, 473, 447, 443, 291, 318, 336, 372, 377],
        "acc_open_2022": [486, 370, 256, 487, 372, 166, 157, 345, 282, 372],
        "acc_open_2023": [187, 199, 151, 369, 459, 459, 204, 338, 316, 319],
        "growth_rate": [0.12, 0.08, 0.19, 0.19, 0.12, 0.15, 0.14, 0.09, 0.19,0.11]


    },
    
    "Post Office Saving Account": {
        "region_name": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli", "Vellore", "Erode", "Dindigul", "Kancheepuram"],
        "acc_open_2019": [387, 199, 304, 268, 305, 462, 250, 370, 396, 486],
        "acc_open_2020": [287, 453, 440, 346, 204, 334, 325, 277, 278, 229],
        "acc_open_2021": [409, 315, 310, 263, 475, 276, 367, 449, 369, 301],
        "acc_open_2022": [289, 413, 488, 306, 385, 319, 420, 295, 467, 493],
        "acc_open_2023": [457, 259, 470, 432, 421, 287, 325, 296, 396, 235],
        "growth_rate": [0.18, 0.14, 0.16, 0.12, 0.08, 0.11, 0.07, 0.19, 0.15,0.13]
        },
   
    "Post Office Saving Account": {
        "region_name": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli", "Vellore", "Erode", "Dindigul", "Kancheepuram"],
        "acc_open_2019": [387, 199, 304, 268, 305, 462, 250, 370, 396, 486],
        "acc_open_2020": [287, 453, 440, 346, 204, 334, 325, 277, 278, 229],
        "acc_open_2021": [409, 315, 310, 263, 475, 276, 367, 449, 369, 301],
        "acc_open_2022": [289, 413, 488, 306, 385, 319, 420, 295, 467, 493],
        "acc_open_2023": [457, 259, 470, 432, 421, 287, 325, 296, 396, 235],
        "growth_rate": [0.18, 0.14, 0.16, 0.12, 0.08, 0.11, 0.07, 0.19, 0.15,0.13]
    }

}
# Sample Data: Scheme Features
schemes = [
    {
        "scheme_name": "Sukanya Samriddhi Yojana",
        "scheme_type": "Savings",
        "target_gender": "All",
        "target_age_group": "18-60",
        "min_investment": 1000,
        "max_investment": 10000,
        "roi": 5.0,
        "risk_level": "Low",
        "target_occupation": "Salaried",
        "target_income_level": "Middle",
        "target_education_level": "Graduate",
        "tax_benefit": "Yes",
        "expected_growth_multiplier": 1.2
    },
    {
        "scheme_name": "Kishan Vikas Patra",
        "scheme_type": "Investment",
        "target_gender": "All",
        "target_age_group": "25-40",
        "min_investment": 5000,
        "max_investment": 20000,
        "roi": 7.5,
        "risk_level": "Medium",
        "target_occupation": "Self-Employed",
        "target_income_level": "Upper Middle",
        "target_education_level": "Post-Graduate",
        "tax_benefit": "No",
        "expected_growth_multiplier": 1.5
    },
    {
        "scheme_name": "Post Office Saving Account",
        "scheme_type": "Fixed Deposit",
        "target_gender": "All",
        "target_age_group": "30-50",
        "min_investment": 10000,
        "max_investment": 50000,
        "roi": 6.5,
        "risk_level": "Low",
        "target_occupation": "Business",
        "target_income_level": "High",
        "target_education_level": "Graduate",
        "tax_benefit": "Yes",
        "expected_growth_multiplier":1.3
        }
]
# Preprocess Region Data
region_df = pd.DataFrame(data["Kishan Vikas Patra"])
features = ["acc_open_2023", "acc_open_2022", "acc_open_2021", "acc_open_2020", "acc_open_2019", "growth_rate"]
numerical_features = features
scaler = MinMaxScaler()
region_df[numerical_features] = scaler.fit_transform(region_df[numerical_features])

# Create a Region-Scheme Matrix
scheme_df = pd.DataFrame(schemes)

# Encoding categorical features for schemes
encoded_scheme_features = pd.get_dummies(
    scheme_df[
        ["scheme_type", "target_gender", "risk_level", "tax_benefit", "target_occupation"]
    ]
)

# Scaling numerical features in schemes
scheme_numerical = scheme_df[["min_investment", "max_investment", "roi", "expected_growth_multiplier"]]
scaled_scheme_numerical = scaler.fit_transform(scheme_numerical)

# Combine encoded categorical and scaled numerical features
scheme_features = np.hstack([encoded_scheme_features, scaled_scheme_numerical])

# Pad region_features to match scheme_features dimensions
region_features = region_df[numerical_features].values
padded_region_features = np.pad(region_features, ((0, 0), (0, scheme_features.shape[1] - region_features.shape[1])), mode='constant')

# Calculate Region-Scheme Similarity
similarity_matrix = cosine_similarity(padded_region_features, scheme_features)

def predict_accounts_and_probability(region_name, scheme_name):
    if region_name not in data[scheme_name]["region_name"] or scheme_name not in scheme_df["scheme_name"].values:
        return None
    
    region_idx = data[scheme_name]["region_name"].index(region_name)
    scheme_idx = scheme_df[scheme_df["scheme_name"] == scheme_name].index[0]
    
    # Get current year's accounts
    current_accounts = data[scheme_name]["acc_open_2023"][region_idx]
    
    # Get region's growth rate
    growth_rate = data[scheme_name]["growth_rate"][region_idx]
    
    # Get scheme's growth multiplier
    scheme_multiplier = schemes[scheme_idx]["expected_growth_multiplier"]
    
    # Calculate predicted accounts for next year
    predicted_accounts = int(current_accounts * (1 + growth_rate) * scheme_multiplier)
    
    # Calculate probability score
    raw_score = similarity_matrix[region_idx, scheme_idx]
    exp_score = np.exp(raw_score)
    sum_exp_scores = np.sum(np.exp(similarity_matrix[region_idx]))
    probability = exp_score / sum_exp_scores
    
    return {
        "region": region_name,
        "scheme": scheme_name,
        "current_accounts": current_accounts,
        "predicted_accounts_next_year": predicted_accounts,
        "probability_score": float(probability),
        "probability_percentage": f"{probability * 100:.2f}%"
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        region_name = data.get('region')
        scheme_name = data.get('scheme')
        
        if not region_name or not scheme_name:
            return jsonify({"error": "Missing region or scheme name"}), 400
            
        prediction = predict_accounts_and_probability(region_name, scheme_name)
        
        if prediction is None:
            return jsonify({"error": "Invalid region or scheme name"}), 400
            
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
