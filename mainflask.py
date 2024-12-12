from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
import tensorflow as tf
import google.generativeai as genai
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

# New Filters DataFrame
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

user_feedback_df = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'location': ['Theni', 'Chennai', 'Thiruvallur', 'Coimbatore', 'Ariyalur'],
    'scheme':  ['Sukanya Samriddhi Yojana', 'Senior citizen saving schemes', 'Kisan Vikas Patra', 'Post office saving account', 'Monthly income scheme'],
    'rating': [4.5, 3.8, 4.2, 3.9, 4.0]
})

# Data Preprocessing
scaler = MinMaxScaler()
location_df[['population_density', 'income_level']] = scaler.fit_transform(
    location_df[['population_density', 'income_level']]
)

# Encoding for Location Features
location_encoder = OneHotEncoder(sparse_output=False)
location_cat_encoded = location_encoder.fit_transform(location_df[['farming_cycle', 'seasonal_pattern']])
location_cat_cols = location_encoder.get_feature_names_out(['farming_cycle', 'seasonal_pattern'])
location_encoded_df = pd.DataFrame(location_cat_encoded, columns=location_cat_cols)

# Encoding for Scheme Features
scheme_encoder = OneHotEncoder(sparse_output=False)
scheme_cat_encoded = scheme_encoder.fit_transform(scheme_df[['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre']])
scheme_cat_cols = scheme_encoder.get_feature_names_out(['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre'])
scheme_encoded_df = pd.DataFrame(scheme_cat_encoded, columns=scheme_cat_cols)

# Prepare Features
location_features = pd.concat([
    location_df[['population_density', 'gender_ratio', 'income_level']],
    location_encoded_df
], axis=1)

scheme_features = pd.concat([
    scheme_df[['ROI']],
    scheme_encoded_df
], axis=1)

def create_improved_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train Autoencoders
location_autoencoder = create_improved_autoencoder(location_features.shape[1])
location_autoencoder.fit(location_features, location_features, epochs=20, batch_size=2, verbose=0)

scheme_autoencoder = create_improved_autoencoder(scheme_features.shape[1])
scheme_autoencoder.fit(scheme_features, scheme_features, epochs=20, batch_size=2, verbose=0)

def create_improved_recommendation_model(location_dim, scheme_dim, filter_dim):
    location_input = Input(shape=(location_dim,))
    scheme_input = Input(shape=(scheme_dim,))
    filter_input = Input(shape=(filter_dim,))

    # Location processing
    x1 = Dense(32, activation='relu')(location_input)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    # Scheme processing
    x2 = Dense(32, activation='relu')(scheme_input)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)

    # Filter processing
    x3 = Dense(16, activation='relu')(filter_input)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.2)(x3)

    # Concatenate all inputs
    concat = Concatenate()([x1, x2, x3])
    x = Dense(64, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[location_input, scheme_input, filter_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare training data
X_loc = location_features.values
X_scheme = scheme_features.values

# Create filter encoding helper
def encode_filters(filters):
    # Create a zero-filled filter feature vector
    filter_features = np.zeros((len(scheme_df), len(filters_df)))
    
    for i, row in filters_df.iterrows():
        category = row['filter_category']
        if category in filters:
            filter_val = filters.get(category)
            if filter_val in row['filter_values']:
                # Mark the corresponding filter value
                col_idx = row['filter_values'].index(filter_val)
                filter_features[:, i] = col_idx + 1  # Avoid zero
    
    return filter_features

# Prepare training labels (using complex weighting)
def prepare_training_labels():
    y = []
    for i in range(len(location_df)):
        loc_schemes = []
        for j in range(len(scheme_df)):
            # Use existing ratings or create synthetic rating logic
            base_rating = user_feedback_df[
                (user_feedback_df['location'] == location_df.loc[i, 'location']) & 
                (user_feedback_df['scheme'] == scheme_df.loc[j, 'scheme'])
            ]['rating'].values

            # If no existing rating, generate a synthetic one
            rating = base_rating[0] / 5.0 if len(base_rating) > 0 else 0.5
            loc_schemes.append(rating)
        y.extend(loc_schemes)
    
    return np.array(y)

# Prepare extended training data
y_train = prepare_training_labels()
X_loc_extended = np.repeat(X_loc, len(X_scheme), axis=0)
X_scheme_extended = np.tile(X_scheme, (len(X_loc), 1))

# Create filter features (initial simple version)
X_filter = encode_filters({})
# Extend X_filter to match other dimensions
X_filter_extended = np.tile(X_filter, (len(X_loc), 1))

# Train recommendation model with corrected dimensions
recommendation_model = create_improved_recommendation_model(
    X_loc.shape[1], X_scheme.shape[1], X_filter.shape[1]
)
recommendation_model.fit(
    [X_loc_extended, X_scheme_extended, X_filter_extended],  # Updated X_filter to X_filter_extended
    y_train, 
    epochs=20, 
    batch_size=2, 
    verbose=0
)
def generate_filter_matrix(location, filters=None):
    """
    Generate a matrix of recommendations based on location and optional filters
    """
    # Validate location
    if location not in location_df['location'].values:
        raise ValueError(f"Invalid location. Choose from: {', '.join(location_df['location'])}")
    
    # Prepare location features
    loc_idx = location_df.index[location_df['location'] == location].tolist()[0]
    loc_features = location_features.iloc[loc_idx:loc_idx+1].values
    loc_repeated = np.tile(loc_features, (len(scheme_features), 1))
    
    # Prepare filter features
    filter_features = encode_filters(filters) if filters else encode_filters({})
    
    # Make predictions
    predictions = recommendation_model.predict(
        [loc_repeated, scheme_features.values, filter_features], 
        verbose=0
    ).flatten()
    
    # Prepare filter matrix
    filter_matrix = []
    filter_categories = filters_df['filter_category'].tolist()
    
    # If no filters specified, return basic recommendations
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
    
    # Generate filter matrix if filters are specified
    for filter_category in filter_categories:
        if filter_category in filters:
            filter_val = filters.get(filter_category)
            filter_matrix_row = []
            
            for scheme in scheme_df['scheme']:
                # Check if scheme matches the filter
                scheme_filter_val = scheme_df[scheme_df['scheme'] == scheme][
                    filter_category.lower().replace(' ', '_')
                ].values[0]
                
                if scheme_filter_val == filter_val:
                    # If scheme matches filter, use its prediction score
                    score_idx = scheme_df[scheme_df['scheme'] == scheme].index[0]
                    score = predictions[score_idx]
                else:
                    # If scheme doesn't match, give a low score
                    score = 0.1
                
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
        "recommendations": sorted(
            [{
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
        data = request.json
        user_id = data.get('userId')
        location = data.get('location')
        filter_category = data.get('filterCategory')  # Single filter category
        filter_value = data.get('filterValue')      # Single filter value
        
        # Create filters dict with single filter if provided
        filters = {}
        if filter_category and filter_value:
            # Validate filter category exists
            if filter_category in filters_df['filter_category'].values:
                # Validate filter value is valid for the category
                valid_values = filters_df[filters_df['filter_category'] == filter_category]['filter_values'].iloc[0]
                if filter_value in valid_values:
                    filters[filter_category] = filter_value
        
        # Generate recommendations with single filter
        result = generate_filter_matrix(location, filters if filters else None)
        
        return jsonify({
            "status": "success",
            "userId": user_id,
            "data": result
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    




if __name__ == '__main__':
    app.run(debug=True)
