from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
import tensorflow as tf
import google.generativeai as genai
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class DataLoader:
    @staticmethod
    def load_data():
        location_df = pd.DataFrame({
            'location': ['Theni', 'Chennai', 'Thiruvallur', 'Coimbatore', 'Ariyalur'],
            'population_density': [21000, 11000, 8400, 7100, 6900],
            'gender_ratio': [0.92, 0.87, 0.95, 0.98, 0.89],
            'income_level': [45000, 52000, 48000, 38000, 35000],
            'farming_cycle': ['Kharif', 'Rabi', 'Mixed', 'Kharif', 'Mixed'],
            'seasonal_pattern': ['Monsoon', 'Winter', 'Year-round', 'Monsoon', 'Year-round'],
            'education_level': ['Graduate', 'Post Graduate', 'High School', 'Graduate', 'High School'],
            'age_distribution': ['25-35', '35-45', '45-55', '25-35', '35-45'],
            'occupation': ['Farmer', 'IT Professional', 'Business', 'Teacher', 'Farmer']
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

        user_feedback_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'location': ['Theni', 'Chennai', 'Thiruvallur', 'Coimbatore', 'Ariyalur'],
            'scheme': ['Sukanya Samriddhi Yojana', 'Senior citizen saving schemes', 'Kisan Vikas Patra', 'Post office saving account', 'Monthly income scheme'],
            'rating': [4.5, 3.8, 4.2, 3.9, 4.0]
        })

        filters_df = pd.DataFrame({
            'filter_category': ['Gender', 'Age Group', 'Income Level', 'Tax Benefit', 'Risk Level', 'Education Level', 'Occupation'],
            'filter_values': [
                ['Male', 'Female', 'Both'],
                ['Young', 'Adult', 'Senior'],
                ['Low', 'Medium', 'High'],
                ['Yes', 'No'],
                ['Low', 'Medium', 'High'],
                ['High School', 'Graduate', 'Post Graduate'],
                ['Farmer', 'IT Professional', 'Business', 'Teacher']
            ]
        })

        month_df = pd.DataFrame({
            'area_name': ['Central', 'North', 'South', 'West', 'East'],
            'festive_season_month': [[10, 11], [3, 4, 10, 11], [1, 2, 9, 10], [3, 4, 10, 11], [3, 4, 10, 11]],
            'wedding_season_month': [[11, 12, 1, 2], [10, 11, 12, 1, 2], [11, 12, 1, 2, 3], [11, 12, 1, 2], [11, 12, 1, 2]],
            'admission_season_month': [[5, 6, 7], [4, 5, 6], [5, 6, 7], [5, 6, 7], [5, 6, 7]]
        })

        return location_df, scheme_df, user_feedback_df, filters_df, month_df

class DataPreprocessor:
    def __init__(self, location_df, scheme_df):
        self.location_df = location_df
        self.scheme_df = scheme_df
        self.scaler = MinMaxScaler()
        self.location_encoder = OneHotEncoder(sparse_output=False)
        self.scheme_encoder = OneHotEncoder(sparse_output=False)

    def preprocess_data(self):
        # Scale numerical features
        self.location_df[['population_density', 'income_level']] = self.scaler.fit_transform(
            self.location_df[['population_density', 'income_level']]
        )

        # Encode categorical features
        location_cat_encoded = self.location_encoder.fit_transform(
            self.location_df[['farming_cycle', 'seasonal_pattern', 'education_level', 'age_distribution', 'occupation']]
        )
        location_cat_cols = self.location_encoder.get_feature_names_out(
            ['farming_cycle', 'seasonal_pattern', 'education_level', 'age_distribution', 'occupation']
        )
        location_encoded_df = pd.DataFrame(location_cat_encoded, columns=location_cat_cols)

        scheme_cat_encoded = self.scheme_encoder.fit_transform(
            self.scheme_df[['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre']]
        )
        scheme_cat_cols = self.scheme_encoder.get_feature_names_out(
            ['target_gender', 'target_age_group', 'tax_benefit', 'risk_level', 'genre']
        )
        scheme_encoded_df = pd.DataFrame(scheme_cat_encoded, columns=scheme_cat_cols)

        # Combine features
        location_features = pd.concat([
            self.location_df[['population_density', 'gender_ratio', 'income_level']],
            location_encoded_df
        ], axis=1)

        scheme_features = pd.concat([
            self.scheme_df[['ROI']],
            scheme_encoded_df
        ], axis=1)

        return location_features, scheme_features

class ModelBuilder:
    @staticmethod
    def create_autoencoder(input_dim):
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

    @staticmethod
    def create_recommendation_model(location_dim, scheme_dim, filter_dim):
        location_input = Input(shape=(location_dim,))
        scheme_input = Input(shape=(scheme_dim,))
        filter_input = Input(shape=(filter_dim,))

        x1 = Dense(32, activation='relu')(location_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)

        x2 = Dense(32, activation='relu')(scheme_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)

        x3 = Dense(16, activation='relu')(filter_input)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.2)(x3)

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

class RecommendationService:
    def __init__(self, location_features, scheme_features, filters_df):
        self.location_features = location_features
        self.scheme_features = scheme_features
        self.filters_df = filters_df
        self.recommendation_model = None

    def train_model(self, X_loc_extended, X_scheme_extended, X_filter_extended, y_train):
        self.recommendation_model = ModelBuilder.create_recommendation_model(
            self.location_features.shape[1],
            self.scheme_features.shape[1],
            len(self.filters_df)
        )
        self.recommendation_model.fit(
            [X_loc_extended, X_scheme_extended, X_filter_extended],
            y_train,
            epochs=20,
            batch_size=2,
            verbose=0
        )

    def generate_recommendations(self, location, filters=None):
        # Implementation of generate_filter_matrix logic here
        pass

class MonthRecommendationService:
    def __init__(self, month_df, scheme_df):
        self.month_df = month_df
        self.scheme_df = scheme_df

    def get_best_time(self, area_name, scheme_name):
        # Implementation of besttime logic here
        pass

# API Routes
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        recommendation_service = RecommendationService(location_features, scheme_features, filters_df)
        result = recommendation_service.generate_recommendations(
            data.get('location'),
            data.get('filters')
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/monthrec', methods=['POST'])
def month_recommendation():
    try:
        data = request.json
        month_service = MonthRecommendationService(month_df, scheme_df)
        result = month_service.get_best_time(
            data.get('area'),
            data.get('scheme')
        )
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Load and preprocess data
    location_df, scheme_df, user_feedback_df, filters_df, month_df = DataLoader.load_data()
    preprocessor = DataPreprocessor(location_df, scheme_df)
    location_features, scheme_features = preprocessor.preprocess_data()
    
    app.run(debug=True)
