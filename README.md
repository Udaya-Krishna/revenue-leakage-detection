# AI Revenue Leakage Detection System

A Flask-based web application that uses machine learning to detect revenue leakages and anomalies in supermarket transaction data.

## Features

- **File Upload**: Upload CSV files through a modern web interface
- **ML Processing**: Uses pre-trained XGBoost model for anomaly detection
- **Visualization**: Interactive charts showing leakage distribution and anomaly types
- **Multiple Downloads**: Get results as complete dataset, anomalies only, or clean records
- **Real-time Processing**: Process files and get results instantly

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Access the Web Interface**
   - Open your browser and go to `http://localhost:5000`
   - Upload a CSV file with transaction data
   - View results and download processed files

## File Structure

```
ai-rev-leak/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Web interface
├── model/
│   └── super_market/
│       ├── saved_models/          # Pre-trained ML models
│       │   ├── trained_pipeline.pkl
│       │   ├── leakage_encoder.pkl
│       │   └── anomaly_encoder.pkl
│       └── models/
│           └── modelwith_input.ipynb  # Original notebook
├── uploads/                       # Uploaded files (created automatically)
└── outputs/                       # Processed results (created automatically)
```

## Expected CSV Format

Your input CSV should contain columns like:
- Invoice_Number
- Actual_Amount
- Tax_Amount
- Service_Charge
- Discount_Amount
- Other transaction-related fields

## Output

The system generates three types of output files:
1. **Complete Results**: All records with predictions
2. **Anomalies Only**: Records flagged as potential revenue leakage
3. **Clean Records**: Records with no anomalies detected

## Model Information

- **Algorithm**: XGBoost Multi-Output Classifier
- **Targets**: Leakage_Flag and Anomaly_Type
- **Preprocessing**: OneHot encoding for categorical, StandardScaler for numerical
- **Performance**: ~95% accuracy on test data

## Anomaly Types Detected

- Duplicate Entries
- Excess Payment
- Missing Charges
- Payment Status Mismatch
- Under Payment
