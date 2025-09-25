# E-commerce Clickstream Analysis & Prediction

## 1. Project Overview

This project develops an intelligent web application to analyze e-commerce clickstream data. The primary goal is to leverage machine learning to derive actionable insights, enhance customer engagement, and drive sales.

The current version of the application focuses on a **multi-class classification problem**: predicting the final page category a user will visit based on their session information.

## 2. Features

- **Interactive Streamlit Dashboard**: User-friendly web interface for real-time analysis.
- **File Upload**: Users can upload their own test data in CSV format.
- **Real-time Prediction**: The application trains a model on historical data and provides instant predictions for the uploaded data.
- **Data Visualization**: Includes visualizations for data summaries and prediction distributions.

## 3. Setup and Installation

Follow these steps to run the application locally:

**Prerequisites:**
- Python 3.8+
- pip (Python package installer)

**Steps:**

1.  **Clone the repository (or create the project folder):**
    ```bash
    git clone <your-repo-url>
    cd clickstream_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your data:**
    - Make sure `train.csv` and `test.csv` are in the root of the project folder.

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## 4. Methodology

- **Data Preprocessing**:
  - Missing values are imputed using median (for numerical) and mode (for categorical).
  - Categorical features are one-hot encoded.
  - Numerical features are standardized using `StandardScaler`.
- **Modeling**:
  - An **XGBoost Classifier** is used for the multi-class classification task.
  - The entire workflow (preprocessing + modeling) is encapsulated in a Scikit-learn Pipeline for robustness and reproducibility.
- **Evaluation**:
  - The model's performance is visualized by showing the distribution of predicted page categories.

## 5. Folder Structure

```
clickstream_project/
├── .gitignore          # Files to be ignored by Git
├── app.py              # Main Streamlit application script
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── tests/
    └── test_preprocessing.py # Unit tests
```