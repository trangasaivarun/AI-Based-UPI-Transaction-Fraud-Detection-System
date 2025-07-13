import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import time
import base64
import pickle 
# import subprocess
# subprocess.check_call(["pip", "install", "xgboost"])
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="AI-Based UPI Transaction Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 16px;">
        <img src="https://img.icons8.com/ios-filled/100/lock--v1.png" width="60"/>
        <h1 style="margin-bottom: 0;">UPI Fraud Detection</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
This application helps detect fraudulent UPI transactions using machine learning.
You can either:
* Enter a single transaction's details
* Upload a CSV file with multiple transactions
""")

# Load the model
@st.cache_resource
def load_model():
    try:
        with open("UPI Fraud Detection Final.pkl", 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

loaded_model = load_model()

# Define the expected feature lists
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 
      'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 
      'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 
      'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 
      'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']



def preprocess_for_prediction(df):
    """
    Preprocess new data to match the training data format
    """
    # Initialize the output DataFrame with zeros for all features
    all_columns = ['amount', 'Year', 'Month']
    
    # Add Transaction_Type columns
    all_columns.extend([f'Transaction_Type_{type}' for type in tt])
    
    # Add Payment_Gateway columns
    all_columns.extend([f'Payment_Gateway_{gateway}' for gateway in pg])
    
    
    # Add Transaction_State columns
    all_columns.extend([f'Transaction_State_{state}' for state in ts])
    
    # Add Merchant_Category columns
    all_columns.extend([f'Merchant_Category_{cat}' for cat in mc])
    
    # Create DataFrame with zeros
    output = pd.DataFrame(0, index=df.index, columns=all_columns)
    
    # Copy numeric values
    output['amount'] = df['amount']
    output['Year'] = df['Year']
    output['Month'] = df['Month']
    
    # Set categorical variables
    for idx, row in df.iterrows():
        # Transaction Type
        col_name = f'Transaction_Type_{row["Transaction_Type"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
            
        # Payment Gateway
        col_name = f'Payment_Gateway_{row["Payment_Gateway"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
         
            
        # Transaction State
        col_name = f'Transaction_State_{row["Transaction_State"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
            
        # Merchant Category
        col_name = f'Merchant_Category_{row["Merchant_Category"]}'
        if col_name in output.columns:
            output.at[idx, col_name] = 1
    
    return output

def validate_input(amount, date):
    if amount <= 0:
        st.error("Amount must be greater than 0")
        return False
    if date > datetime.datetime.now().date():
        st.error("Transaction date cannot be in the future")
        return False
    return True

def display_prediction(prediction, probability):
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
        st.warning(f"Fraud Probability: {probability[0][1]:.2%}")
    else:
        st.success("✅ Legitimate Transaction")
        st.info(f"Fraud Probability: {probability[0][1]:.2%}")

def display_statistics(amount, merchant_cat, state):
    st.subheader("Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Format large numbers with commas
        formatted_amount = "{:,.2f}".format(amount)
        st.metric("Amount", f"₹{formatted_amount}")
    with col2:
        st.metric("Category", merchant_cat)
    with col3:
        st.metric("Location", state)

def process_batch_file(df):
    try:
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract Year and Month
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Preprocess the data
        processed_df = preprocess_for_prediction(df)
        
        if loaded_model is None:
            st.error("Model not loaded properly")
            return None
            
        # Make predictions
        predictions = loaded_model.predict(processed_df)
        probabilities = loaded_model.predict_proba(processed_df)
        
        # Add predictions to original dataframe
        df['Predicted_Fraud'] = predictions
        df['Fraud_Probability'] = probabilities[:, 1]
        
        return df
        
    except Exception as e:
        st.error(f"Error in process_batch_file: {str(e)}")
        return None

# Sidebar for navigation
page = st.sidebar.selectbox("Choose Input Method", ["Single Transaction", "Batch Processing"])

if page == "Single Transaction":
    st.header("Enter Transaction Details")
    
    with st.form("single_txn_form"):
        st.markdown("#### 📝 Enter Transaction Details")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=0.0, value=100.0, step=100.0, format="%.2f", help="Enter the transaction amount in INR.")
            transaction_type = st.selectbox("🔄 Transaction Type", tt, help="Select the type of transaction.")
            payment_gateway = st.selectbox("🏦 Payment Gateway", pg, help="Select the payment gateway used.")
        with col2:
            transaction_state = st.selectbox("📍 Transaction State", ts, help="Select the state where the transaction occurred.")
            merchant_category = st.selectbox("🛒 Merchant Category", mc, help="Select the merchant category.")
            transaction_date = st.date_input("📅 Transaction Date", datetime.datetime.now(), help="Select the date of the transaction.")
        submitted = st.form_submit_button("🚦 Check Transaction")
        if submitted:
            if validate_input(amount, transaction_date):
                # Create input dataframe
                input_df = pd.DataFrame({
                    'amount': [amount],
                    'Transaction_Type': [transaction_type],
                    'Payment_Gateway': [payment_gateway],
                    'Transaction_State': [transaction_state],
              
                    'Merchant_Category': [merchant_category],
                    'Year': [transaction_date.year],
                    'Month': [transaction_date.month]
                })
                
                # Preprocess data
                processed_data = preprocess_for_prediction(input_df)
                
                try:
                    # Make prediction
                    prediction = loaded_model.predict(processed_data)
                    probability = loaded_model.predict_proba(processed_data)
                    
                    # Display results
                    display_statistics(amount, merchant_category, transaction_state)
                    display_prediction(prediction[0], probability)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"Processed data shape: {processed_data.shape}")
                    st.write(f"Processed columns: {processed_data.columns.tolist()}")

else:
    st.header("Upload CSV File for Batch Processing")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show sample of uploaded data
            st.subheader("Sample of Uploaded Data")
            st.write(df.head())
            
            if st.button("Process Transactions"):
                # Process the file
                results = process_batch_file(df)
                
                if results is not None:
                    # Display results
                    st.subheader("Results")
                    st.write(f"Total Transactions: {len(results)}")
                    st.write(f"Flagged as Fraud: {sum(results['Predicted_Fraud'])}")
                    
                    # Show fraud distribution
                    st.subheader("Fraud Distribution")
                    fraud_counts = results['Predicted_Fraud'].value_counts()
                    st.bar_chart(fraud_counts)
                    
                    # Display detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="fraud_detection_results.csv">Download Results CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
        st.markdown("""
        ### Expected CSV Format
        Your CSV file should contain the following columns:
        - Date (DD-MM-YYYY format)
        - amount
        - Transaction_Type
        - Payment_Gateway
        - Transaction_State
        
        - Merchant_Category
        """)

# Footer
st.markdown("---")
st.markdown("""
### About this App
This UPI Fraud Detection system uses machine learning to analyze transaction patterns and identify potential fraud.
The model considers various factors including:
- Transaction amount
- Transaction type
- Payment gateway
- Geographic location
- Merchant category

### How to Use
1. Choose between single transaction or batch processing
2. Enter transaction details or upload a CSV file
3. Click the check/process button to get predictions
4. Review the results and download if needed
""")

st.markdown(
    "**Note:** This application is intended solely for research and demonstration purposes. The dataset utilized for model training and evaluation was obtained from Kaggle."
)

st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <h4>Developed by</h4>
        <div style="display: flex; justify-content: center; gap: 32px; flex-wrap: wrap;">
            <div style="border: 1px solid #eee; border-radius: 10px; padding: 16px 24px; min-width: 200px;">
                <b>Talluri Ranga Sai Varun</b><br>
                <a href="https://www.linkedin.com/in/ranga-sai-varun-talluri-059b5b275/" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="28"/>
                </a>
            </div>
            <div style="border: 1px solid #eee; border-radius: 10px; padding: 16px 24px; min-width: 200px;">
                <b>Telagamsetty Viswajith Gupta</b><br>
                <a href="https://www.linkedin.com/in/viswajith-gupta-telagamsetty-1b71812a2/" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="28"/>
                </a>
            </div>
            <div style="border: 1px solid #eee; border-radius: 10px; padding: 16px 24px; min-width: 200px;">
                <b>Dokala Manoj Kumar</b><br>
                <a href="https://www.linkedin.com/in/manoj-dokala-37932b2b2/" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="28"/>
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
