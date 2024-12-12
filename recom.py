from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization

# Initialize Flask app
app = Flask(__name__)

# Data Initialization
location_df = pd.DataFrame({
    'location': ['Theni', 'Chennai', 'Thiruvallur', 'Coimbatore', 'Ariyalur'],
    'population_density': [21000, 11000, 8400, 7100, 6900],
    'gender_ratio': [0.92, 0.87, 0.95, 0.98, 0.89],
    'income_level': [45000, 52000, 48000, 38000, 35000],
    'farming_cycle': ['Kharif', 'Rabi', 'Mixed', 'Kharif', 'Mixed'],
    'seasonal_pattern': ['Monsoon', 'Winter', 'Year-round', 'Monsoon', 'Year-round']
})

scheme_df = pd.DataFrame({
    'scheme': ['Sukanya Samriddhi Yojana', 'Senior citizen saving schemes', 'Kisan Vikas Patra', 'Post office saving account', 'Monthly income scheme'],
    'ROI': [12.5, 8.2, 15.3, 10.1, 9.8],
    'target_gender': ['Female', 'Male', 'Both', 'Female', 'Both'],
    'target_age_group': ['Adult', 'Senior', 'Adult', 'Senior', 'Young'],
    'tax_benefit': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'risk_level': ['Low', 'Medium', 'High', 'Low', 'Medium'],
    'genre': ['Savings', 'Senior', 'Agriculture', 'General', 'Income']
})

filters_df = pd.DataFrame({
    'filter_category': ['Gender', 'Age Group', 'Income Level', 'Tax Benefit', 'Risk Level'],
    'filter_values': [
        ['Male', 'Female', 'Both'],
        ['Young', 'Adult', 'Senior'],
        ['Low', 'Medium', 'High'],
        ['Yes', 'No'],
        ['Low', 'Medium', 'High']
    ]
})

# Data Preprocessing
scaler = MinMaxScaler()
location_df[['population_density', 'income_level']] = scaler.fit_transform(
    location_df[['population_density', 'income_level']]
)

location_encoder = OneHotEncoder(sparse_output=False)
location_cat_encoded = location_encoder.fit_transform(location_df[['farming_cycle', 'seasonal_pattern']])
location_cat_cols = location_encoder.get_feature_names_out(['farming_cycle', 'seasonal_pattern'])
location_encoded_df = pd.DataFrame(location_cat_encoded, columns=location_cat_cols)

scheme_encoder = OneHotEncoder(sparse_output=False)
scheme_cat_encoded = scheme_encoder.fit_transform(scheme_df[['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre']])
scheme_cat_cols = scheme_encoder.get_feature_names_out(['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre'])
scheme_encoded_df = pd.DataFrame(scheme_cat_encoded, columns=scheme_cat_cols)

location_features = pd.concat([
    location_df[['population_density', 'gender_ratio', 'income_level']],
    location_encoded_df
], axis=1)

scheme_features = pd.concat([
    scheme_df[['ROI']],
    scheme_encoded_df
], axis=1)

def encode_filters(filters):
    filter_features = np.zeros((len(scheme_df), len(filters_df)))
    for i, row in filters_df.iterrows():
        category = row['filter_category']
        if category in filters:
            filter_val = filters.get(category)
            if filter_val in row['filter_values']:
                col_idx = row['filter_values'].index(filter_val)
                filter_features[:, i] = col_idx + 1
    return filter_features

def generate_filter_matrix(location, filters=None):
    if location not in location_df['location'].values:
        raise ValueError(f"Invalid location. Choose from: {', '.join(location_df['location'])}")

    loc_idx = location_df.index[location_df['location'] == location].tolist()[0]
    loc_features = location_features.iloc[loc_idx:loc_idx+1].values
    loc_repeated = np.tile(loc_features, (len(scheme_features), 1))

    filter_features = encode_filters(filters) if filters else encode_filters({})

    predictions = recommendation_model.predict(
        [loc_repeated, scheme_features.values, filter_features], 
        verbose=0
    ).flatten()

    if not filters:
        recommendations = [{
            "scheme": scheme,
            "score": float(score),
            "details": {
                "ROI": float(scheme_df[scheme_df['scheme'] == scheme]['ROI'].values[0]),
                "target_gender": scheme_df[scheme_df['scheme'] == scheme]['target_gender'].values[0],
                "risk_level": scheme_df[scheme_df['scheme'] == scheme]['risk_level'].values[0],
                "genre": scheme_df[scheme_df['scheme'] == scheme]['genre'].values[0]
            }
        } for scheme, score in zip(scheme_df['scheme'], predictions)]

        return {
            "recommendations": sorted(recommendations, key=lambda x: x['score'], reverse=True)[:3],
            "filter_matrix": None
        }

    filter_matrix = []
    for filter_category in filters_df['filter_category']:
        if filter_category in filters:
            filter_val = filters[filter_category]
            filter_matrix_row = []

            for scheme in scheme_df['scheme']:
                scheme_filter_val = scheme_df[scheme_df['scheme'] == scheme][
                    filter_category.lower().replace(' ', '_')
                ].values[0]

                score_idx = scheme_df[scheme_df['scheme'] == scheme].index[0]
                score = predictions[score_idx] if scheme_filter_val == filter_val else 0.1

                filter_matrix_row.append({
                    "scheme": scheme,
                    "score": float(score),
                    "matches_filter": scheme_filter_val == filter_val
                })

            filter_matrix.append({
                "filter_category": filter_category,
                "filter_value": filter_val,
                "schemes": filter_matrix_row
            })

    return {
        "location": location,
        "filters": filters,
        "recommendations": sorted([
            {
                "scheme": scheme_df.loc[i, 'scheme'],
                "score": float(predictions[i]),
                "details": {
                    "ROI": float(scheme_df.loc[i, 'ROI']),
                    "target_gender": scheme_df.loc[i, 'target_gender'],
                    "risk_level": scheme_df.loc[i, 'risk_level'],
                    "genre": scheme_df.loc[i, 'genre']
                }
            } for i in range(len(scheme_df))], 
            key=lambda x: x['score'], 
            reverse=True
        )[:3],
        "filter_matrix": filter_matrix
    }

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        data = request.get_json()
        user_id = data.get('userId')
        location = data.get('location')
        filter_category = data.get('filterCategory', "").capitalize()
        filter_value = data.get('filterValue', "").capitalize()

        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required"}), 400
        if not location:
            return jsonify({"status": "error", "message": "Location is required"}), 400
        if location not in location_df['location'].values:
            return jsonify({
                "status": "error",
                "message": f"Invalid location. Choose from: {', '.join(location_df['location'])}"
            }), 400

        filters = {}
        if filter_category and filter_value:
            if filter_category in filters_df['filter_category'].values:
                valid_values = filters_df[filters_df['filter_category'] == filter_category]['filter_values'].iloc[0]
                if filter_value in valid_values:
                    filters[filter_category] = filter_value
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Invalid filter value for {filter_category}. Valid values: {', '.join(valid_values)}"
                    }), 400
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid filter category. Choose from: {', '.join(filters_df['filter_category'])}"
                }), 400

        result = generate_filter_matrix(location, filters if filters else None)

        if not result.get("recommendations"):
            return jsonify({
                "status": "success",
                "userId": user_id,
                "data": {"message": "No recommendations found for the specified filters"}
            }), 200

        return jsonify({"status": "success", "userId": user_id, "data": result})

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing key in payload: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
