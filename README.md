# 🤖 Data Cleaning Agent

An agentic AI data-cleaning application that uses
LangGraph, LangChain, Ollama, Qwen3, and Pandas to
automatically inspect, clean, validate, and save CSV data.

## 🚀 Overview

The Data Cleaning Agent accepts a CSV dataset and uses
an AI agent to identify and resolve common data-quality
issues.

The agent can:

- Inspect dataset structure
- Detect missing values
- Detect duplicate rows
- Convert numeric columns
- Remove duplicates
- Fill missing values
- Validate the cleaned dataset
- Save the cleaned dataset

## 🏗️ Architecture

User Request
     ↓
LangGraph Agent
     ↓
Qwen3 / Ollama
     ↓
Tool Selection
     ↓
Pandas Data Processing
     ↓
Validation
     ↓
Cleaned CSV

## 🛠️ Technologies

- Python
- Pandas
- LangChain
- LangGraph
- Ollama
- Qwen3

## 🔧 Available Tools

| Tool | Purpose |
|------|---------|
| `inspect_data` | Profiles the dataset |
| `remove_duplicates` | Removes duplicate rows |
| `fill_missing_values` | Handles missing values |
| `fix_numeric_columns` | Converts numeric columns |
| `validate_data` | Checks data quality |
| `save_cleaned_data` | Saves cleaned dataset |

## 🔄 Agent Workflow

1. Load the CSV dataset
2. Inspect the dataset
3. Identify data-quality issues
4. Select appropriate tools
5. Perform cleaning operations
6. Validate the result
7. Save the cleaned dataset
8. Return a summary

## 📊 Example

Input:

`sales.csv`

Output:

`output/cleaned_data.csv`

## 💻 Installation

Clone the repository:

git clone <your-repository-url>

Install dependencies:

pip install -r requirements.txt

Make sure Ollama is installed and the Qwen3 model
is available locally.

## ▶️ Run

python main.py

## 📸 Example Output

![Agent Output](screenshots/agent-output.png)

## 🔮 Future Improvements

- Data-quality scoring
- Automatic schema detection
- Outlier detection
- Data-quality reports
- Support for multiple file formats
- PySpark integration
- Databricks integration
- Automated data pipeline monitoring