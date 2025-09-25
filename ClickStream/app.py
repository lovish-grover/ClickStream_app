# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. App Configuration ---
st.set_page_config(
    page_title="Clothing Model Page Prediction 👗",
    page_icon="📈",
    layout="wide"
)

# --- 2. Caching Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads data from a CSV file and caches it."""
    return pd.read_csv(file_path)

# --- 3. Main App UI ---
st.title("👗 E-commerce Page Category Prediction")
st.write("""
    **Predict which page category a customer session will land on based on session data.**
    This model helps understand user navigation patterns for different clothing models.
""")

# --- 4. Sidebar for User Inputs ---
st.sidebar.header("⚙️ Controls")

# File uploader for test data
uploaded_file = st.sidebar.file_uploader(
    "Upload your test CSV file", type=["csv"]
)

# --- 5. Main Panel Logic ---
# Load the training data once
try:
    train_df = load_data('train.csv')
except FileNotFoundError:
    st.error("`train.csv` not found. Please place `train.csv` in the same folder as this script.")
    st.stop()

if uploaded_file is not None:
    test_df = load_data(uploaded_file)
    
    st.header("1. Exploratory Data Analysis (EDA) on Test Data")
    st.dataframe(test_df.head())
    
    st.header("2. 🚀 Page Category Prediction (Classification)")
    
    # --- Preprocessing and Model Pipeline ---
    # Define features (X) and target (y)
    X_train = train_df.drop(['page', 'session_id'], axis=1)
    y_train = train_df['page']
    
    # Convert labels from [1,2,3,4,5] to [0,1,2,3,4] for the model
    y_train = y_train - 1
    
    # Ensure the test set has the same columns as the training set
    X_test = test_df.drop(['session_id'], axis=1, errors='ignore')
    
    # Align columns
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0 
    X_test = X_test[train_cols]


    # Identify feature types
    numeric_features = ['order', 'price', 'price_2']
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    # Define the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Create the full pipeline
    clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
    
    # Train the pipeline
    with st.spinner('Training model on historical data...'):
        clf_pipeline.fit(X_train, y_train)
    st.success("Model training complete!")

    # Make predictions
    with st.spinner('Making predictions on your data...'):
        predictions = clf_pipeline.predict(X_test)
        # Convert predictions back to original labels [1,2,3,4,5] for display
        predictions_original_labels = predictions + 1
    
    # --- Display Results ---
    st.subheader("Prediction Results")
    
    results_df = X_test.copy()
    results_df['Predicted_Page_Category'] = predictions_original_labels
    
    st.dataframe(results_df)

    # Visualize the prediction distribution
    st.subheader("Distribution of Predicted Page Categories")
    fig, ax = plt.subplots()
    sns.countplot(x='Predicted_Page_Category', data=results_df, ax=ax, order = sorted(results_df['Predicted_Page_Category'].unique()))
    ax.set_title('Count of Predictions per Page Category')
    ax.set_xlabel('Page Category')
    ax.set_ylabel('Count')
    st.pyplot(fig)

else:
    st.info("Awaiting for a CSV file to be uploaded. Please use the sidebar control.")