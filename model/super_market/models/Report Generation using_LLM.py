import pandas as pd
import google.generativeai as genai
import os

def create_revenue_leakage_report(api_key, file_path):
    """
    Analyzes a supermarket dataset for revenue leakage and generates a detailed
    report using a large language model.
    """
    try:
        # Configure the Google Generative AI with your API key
        genai.configure(api_key=api_key)

        # --- Set the conversion rate ---
        USD_TO_INR_RATE = 87.79

        # Load the dataset
        df = pd.read_csv(file_path)

        # Filter for data with 'Anomaly' leakage using the predicted flag
        leakage_data = df[df['Leakage_Flag_Pred'] == 'Anomaly']

        # Take a larger, focused sample to avoid exceeding token limits
        leakage_sample = leakage_data[[
            'Billing_Date',
            'Anomaly_Type_Pred', # Corrected column name
            'Store_Branch',
            'Product_Name',
            'Product_Quantity',
            'Unit_Price',
            'Billed_Amount',
            'Paid_Amount',
            'Balance_Amount',
            'Tax_Amount',
            'Service_Charge',
            'Discount_Amount',
            'Transaction_Type',
            'Mode_of_Payment'
        ]].head(20)

        leakage_sample_str = leakage_sample.to_string(index=False)

        # --- Root Cause Analysis Data ---
        # Use the predicted anomaly type for grouping
        grouped_by_anomaly_branch = leakage_data.groupby(['Anomaly_Type_Pred', 'Store_Branch'])['Balance_Amount'].sum().reset_index()
        grouped_by_anomaly_branch['Balance_Amount_INR'] = grouped_by_anomaly_branch['Balance_Amount'] * USD_TO_INR_RATE
        grouped_by_anomaly_branch_str = grouped_by_anomaly_branch[['Anomaly_Type_Pred', 'Store_Branch', 'Balance_Amount_INR']].to_string(index=False)

        top_products = leakage_data['Product_Name'].value_counts().head(5).to_string()
        top_cashiers = leakage_data['Cashier_ID'].value_counts().head(5).to_string()

        unique_leakage_dates = leakage_data['Billing_Date'].unique()
        formatted_dates = ", ".join(unique_leakage_dates)

        # Calculate key financial metrics in INR
        total_leakage_amount_inr = leakage_data['Balance_Amount'].sum() * USD_TO_INR_RATE
        total_billed_amount = df['Billed_Amount'].sum()
        leakage_percentage = (total_leakage_amount_inr / (total_billed_amount * USD_TO_INR_RATE + 1e-9)) * 100

        total_tax_leakage_inr = leakage_data['Tax_Amount'].sum() * USD_TO_INR_RATE
        total_service_charge_leakage_inr = leakage_data['Service_Charge'].sum() * USD_TO_INR_RATE
        total_discount_leakage_inr = leakage_data['Discount_Amount'].sum() * USD_TO_INR_RATE

        # Add statistical metrics for a more in-depth report in INR
        mean_balance_inr = leakage_data['Balance_Amount'].mean() * USD_TO_INR_RATE
        median_balance_inr = leakage_data['Balance_Amount'].median() * USD_TO_INR_RATE
        std_dev_balance_inr = leakage_data['Balance_Amount'].std() * USD_TO_INR_RATE

        prompt_text = f"""
Perform a comprehensive root cause analysis on the provided supermarket revenue leakage data and create a detailed report. All financial amounts should be in Indian Rupees (INR).

**Report Sections:**
- **Leakage Summary**: Identify the main reasons ('Anomaly_Type_Pred'), locations ('Store_Branch'), and dates of leakage.
- **Root Cause Analysis**: Use the provided grouped data to analyze the root causes of the leakage. Identify which specific anomaly types are most prevalent in which branches. Analyze the top products and cashiers contributing to the leakage to determine if there are specific operational or training issues.
- **Financial Analysis**: Detail the total leaked revenue, leakage percentage, and analyze the specific contributions of taxes, service charges, and discounts to the leakage.
- **Statistical Breakdown**: Include an analysis of the mean, median, and standard deviation of the leaked balance amounts.
- **Actionable Recommendations**: Provide specific, actionable steps to address the identified root causes.

**Data Snapshot:**
- Unique Leakage Dates: {formatted_dates}
- Sample Data:
{leakage_sample_str}

**Root Cause Data (in INR):**
- Total Leakage by Anomaly Type and Branch:
{grouped_by_anomaly_branch_str}
- Top 5 Products with Leakage:
{top_products}
- Top 5 Cashiers with Leakage:
{top_cashiers}

**Key Metrics (in INR):**
- Total Revenue Leaked: ₹{total_leakage_amount_inr:.2f}
- Leakage Percentage: {leakage_percentage:.2f}%
- Total Tax Leakage: ₹{total_tax_leakage_inr:.2f}
- Total Service Charge Leakage: ₹{total_service_charge_leakage_inr:.2f}
- Total Discount Leakage: ₹{total_discount_leakage_inr:.2f}

**Statistical Metrics (Balance Amount) in INR:**
- Mean: ₹{mean_balance_inr:.2f}
- Median: ₹{median_balance_inr:.2f}
- Standard Deviation: ₹{std_dev_balance_inr:.2f}
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_text)
        report_content = response.text
        file_name = "revenue_leakage_report.txt"

        with open(file_name, "w") as f:
            f.write("### AI-Generated Revenue Leakage Report (All amounts in INR)\n\n")
            f.write(report_content)

        print(f"Report successfully saved to {os.path.abspath(file_name)}")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# This is the correct function call.
create_revenue_leakage_report(api_key="*******", file_path="/content/anomaly_data.csv")
