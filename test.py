import time
import pandas as pd
import ollama

csv_file = "telecom_predictions.csv"
chunk_size = 5000
model_name = "llama3"
output_file = "report.txt"


def summarize_chunk(df, i):
    desc = df.describe(include="all", percentiles=[.25, .5, .75]).transpose()
    summary = f"""
    Chunk {i} Summary:
    Shape: {df.shape}
    Columns: {list(df.columns)}
    Stats:
    {desc.to_string()}
    """
    return summary


def ask_llama(prompt):
    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]


def main():
    reports = []
    for i, df in enumerate(pd.read_csv(csv_file, chunksize=chunk_size), start=1):
        print(f"Processing chunk {i}...")
        chunk_summary = summarize_chunk(df, i)
        prompt = f"""
        You are an expert business consultant and data strategist.
        Your task is to analyze the following dataset summary and provide a detailed 
        business-oriented report. Do NOT ask questions back to the user.

        Focus on:
        1. Profitability – where profit is being made, where losses are occurring.
        2. Cost analysis – highlight unnecessary expenses, inefficiencies, or anomalies.
        3. Revenue drivers – identify products, services, or categories contributing the most.
        4. Risk & anomalies – detect unusual trends, missing payments, or outliers.
        5. Actionable recommendations – strategies to reduce losses and maximize profits.

        The dataset may have different column names (e.g., payments, revenue, sales, usage, plans).
        Interpret the data in a business context and provide practical, decision-focused insights.

        Dataset summary:
        {chunk_summary}
        """
        chunk_report = ask_llama(prompt)
        reports.append(f"=== Report for Chunk {i} ===\n{chunk_report}\n")
    print("Generating final summary...")
    final_prompt = f"""
    You are an expert business consultant and strategist. 
    Below are multiple analysis reports generated from different chunks of a large dataset.  

    Your task is to:
    1. Combine them into one unified report.  
    2. Focus on profit vs. loss, revenue drivers, cost leakages, risks, and anomalies.  
    3. Provide clear, actionable recommendations to reduce losses and maximize profits.  
    4. Avoid repeating raw data descriptions; focus on insights that matter to business growth.  
    5. Do NOT ask questions back to the user.  

    Reports:
    {' '.join(reports)}
    """

    final_report = ask_llama(final_prompt)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"Final report saved as {output_file}")


if __name__ == "__main__":
    start = time.time()
    main()
    stop = time.time()
    print(stop-start)
