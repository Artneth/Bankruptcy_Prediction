import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.page_link("https://github.com/Artneth/Bankruptcy_prevention", label="View on GitHub")


# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("random_forest.pkl")  # Load trained model
    scaler = joblib.load("scaler.pkl")  # Load the trained scaler
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("💲Bankruptcy Prediction App💲")

# Expected feature columns
feature_names = ['industrial_risk', 'management_risk', 'financial_flexibility', 
                 'credibility', 'competitiveness', 'operating_risk']

mode = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Batch Prediction"])

# 🔹 SINGLE PREDICTION
if mode == "Single Prediction":
    st.header("Single Prediction")
    user_input = []

    for feature in feature_names:
        value = st.number_input(f"Enter {feature}", min_value=-3.0, step=0.01, format="%.2f")
        user_input.append(value)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)

        # Scale input data
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        result = "Bankrupt" if prediction[0] == 1 else "Not Bankrupt"
        st.success(f"Prediction: {result}")

# 🔹 BATCH PREDICTION
elif mode == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Standardize column names (remove spaces, convert to lowercase)
        data.columns = [col.strip().lower() for col in data.columns]
        required_features = [feat.lower() for feat in feature_names]

        st.write("Detected columns:", data.columns.tolist())  # Debugging information

        # Ensure all required columns exist in the uploaded CSV
        if not all(col in data.columns for col in required_features):
            st.error("The uploaded CSV is missing one or more required feature columns!")
        else:
            # Drop class column if present before scaling and prediction
            if 'class' in data.columns:
                true_labels = data['class']  # Save actual values for evaluation
                data = data.drop(columns=['class'])  # Drop before prediction
            else:
                true_labels = None

            # Scale only the input features
            inputs_scaled = scaler.transform(data[required_features])
            predictions = model.predict(inputs_scaled)

            # 🔥 Reverse Predictions Since They Are Flipped
            data['prediction'] = np.where(predictions == 1, "Not Bankrupt", "Bankrupt")

            st.write("Predictions:")
            st.dataframe(data)
            
            label_mapping = {"bankruptcy": 0, "non-bankruptcy": 1}  # Ensure this matches training encoding
            true_labels = true_labels.map(label_mapping)

            # 🔹 If 'class' column was in the file, evaluate model performance
            if true_labels is not None:
                st.subheader("Model Evaluation")
                
                accuracy = accuracy_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)

                st.write(f"**Accuracy:** {accuracy:.2f}")
                st.write(f"**Precision:** {precision:.2f}")
                st.write(f"**Recall:** {recall:.2f}")
                st.write(f"**F1 Score:** {f1:.2f}")

                # Confusion Matrix Visualization
                cm = confusion_matrix(true_labels, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

            # Allow user to download results
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")




