# 🤖 AutoInsight AI

### Autonomous AI Business Data Analyst

AutoInsight AI is an Agentic AI-powered business data analysis platform that automatically analyzes CSV datasets, cleans data, performs exploratory data analysis, trains machine learning models, evaluates results, reflects on model performance, and performs corrective iterations when required.

It also generates business insights and actionable recommendations from the analyzed data.

## 🚀 Key Features

- 📁 CSV dataset upload
- 🧠 Autonomous agent planning
- 🔍 Automatic dataset analysis
- 🧹 Missing-value handling
- 🧹 Duplicate-row removal
- 📊 Automatic Exploratory Data Analysis (EDA)
- 🤖 Multiple machine learning models
- 👀 Agent observation
- 🧠 Reflection on model performance
- 🔄 Automatic correction and retry
- 🏆 Best-model selection
- 📊 AI Business Insights
- 💡 Business Recommendations

## 🧠 Agent Workflow

Upload CSV
↓
Plan
↓
Analyze
↓
Clean
↓
Auto-EDA
↓
Train Models
↓
Observe
↓
Reflect
↓
Correct / Retry
↓
Complete
↓
Business Insights
↓
Recommendations

## ⭐ What Makes AutoInsight AI Different?

Traditional ML pipelines generally follow:

Data → Train → Evaluate → Finish

AutoInsight AI introduces an agentic feedback loop:

Plan → Execute → Observe → Reflect → Correct → Execute Again

If the initial model performance is weak or models produce similar results, the agent can trigger another model-training iteration using additional model families.

## 🤖 Agent Trace

The system provides a transparent execution trace showing:

GOAL
↓
PLAN
↓
TOOL
↓
OBSERVE
↓
REFLECT
↓
CORRECT
↓
COMPLETE

## 📊 AI Business Insights

After the ML pipeline completes, AutoInsight AI analyzes the cleaned dataset and presents useful business-level findings such as:

- Average numerical values
- Highest and lowest values
- Most common categories
- Important dataset patterns

## 💡 AI Recommendations

The system provides actionable recommendations based on the analyzed data and model performance.

Examples:

- Monitor important business metrics
- Focus on frequently occurring product/customer categories
- Improve features or collect more data when model performance is weak
- Monitor model performance with future datasets

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Google Gemini API

## 🤖 Machine Learning Models

### Baseline Models

- Logistic Regression
- Random Forest

### Correction / Retry Models

- Support Vector Machine (SVM)
- Gradient Boosting
- Corrected Random Forest

## 🧹 Automatic Data Cleaning

The system automatically handles:

- Missing values
- Duplicate rows

Numeric missing values are filled using the median.

Categorical missing values are filled using the mode.

## 📈 Automatic EDA

AutoInsight AI automatically identifies:

- Numeric columns
- Categorical columns
- Numeric distributions
- Feature correlations

Interactive Plotly visualizations are provided for easier analysis.

## 🔐 Gemini Integration

Google Gemini can optionally be used for:

- Agent planning
- Agent reflection
- Retry decisions

Never commit API keys or secrets to GitHub.

## ▶️ Installation

Clone the repository:

git clone https://github.com/rsujithkumar2006-byte/AutoInsight-AI.git

cd AutoInsight-AI

Create virtual environment:

python -m venv venv

Activate:

venv\Scripts\activate

Install dependencies:

pip install -r requirements_upgraded.txt

## ▶️ Run the Application

streamlit run app.py

## 📁 Project Structure

AUTOISIGHT AI/
├── app.py
├── customer_sales.csv
├── requirements_upgraded.txt
├── README.md
└── .gitignore

## 🎯 Use Cases

- 📱 Phone sales analysis
- 🛒 Supermarket sales analysis
- 👥 Customer analytics
- 💰 Revenue analysis
- 📦 Product performance analysis
- 📊 Business decision support

## 🏆 Hackathon

Agentic AI Challenge 2026

Track A — Autonomous Agent Execution

## 👨‍💻 Project

AutoInsight AI

An autonomous business data analyst designed to transform raw business datasets into validated machine-learning results, insights, and actionable recommendations.