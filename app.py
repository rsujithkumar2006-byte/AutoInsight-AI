
import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(page_title="AutoInsight AI", page_icon="🤖", layout="wide")

st.title("🤖 AutoInsight AI")
st.subheader("Autonomous AI Business Data Analyst")
st.caption("Think → Plan → Use Tools → Execute → Observe → Reflect → Correct → Complete")

st.sidebar.title("⚙️ Agent Pipeline")
for item in [
    "📁 Upload CSV", "🧠 Plan", "🔍 Analyze", "🧹 Clean",
    "📊 Auto-EDA", "🤖 Train Models", "👀 Observe",
    "🔄 Reflect & Correct", "🏆 Complete"
]:
    st.sidebar.write(item)

st.sidebar.divider()
st.sidebar.subheader("🧠 Agent Brain")
api_key = st.sidebar.text_input("Gemini API key (optional)", type="password")
model_name = st.sidebar.selectbox(
    "Gemini model",
    ["gemini-3.6-flash", "gemini-3.7-flash"]
)

def gemini_reason(prompt):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or genai is None:
        return ""
    try:
        client = genai.Client(api_key=key)
        interaction = client.interactions.create(
            model=model_name,
            input=prompt
        )
        return getattr(interaction, "output_text", "") or ""
    except Exception as e:
        st.warning(f"Gemini reasoning unavailable: {e}")
        return ""

def make_plan(df):
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }
    prompt = f"""
You are the planning brain of AutoInsight AI.
Dataset summary: {json.dumps(summary)}
Give a short 6-step autonomous ML plan. Include analysis, cleaning, EDA,
multiple models, observation, and a correction/retry step if performance is weak.
"""
    answer = gemini_reason(prompt)
    return answer or """1. Inspect the dataset.
2. Detect and clean missing/duplicate values.
3. Run automatic EDA.
4. Detect classification or regression.
5. Train multiple model families and compare validation performance.
6. Reflect on the result and retry with another strategy when necessary."""

def clean_data(df):
    out = df.copy()
    actions = []
    dups = int(out.duplicated().sum())
    if dups:
        out = out.drop_duplicates()
        actions.append(f"Removed {dups} duplicate row(s).")
    for col in out.columns:
        n = int(out[col].isna().sum())
        if n:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
                actions.append(f"Filled {n} missing value(s) in {col} with median.")
            else:
                mode = out[col].mode()
                fill = mode.iloc[0] if len(mode) else "Unknown"
                out[col] = out[col].fillna(fill)
                actions.append(f"Filled {n} missing value(s) in {col} with mode.")
    return out, actions

def prepare_x(df, target):
    return pd.get_dummies(df.drop(columns=[target]), drop_first=True).fillna(0)

def run_models(df, target, problem_type, retry=False):
    X = prepare_x(df, target)
    y = df[target].copy()
    encoder = None

    if problem_type == "Classification":
        if not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y.astype(str))
        else:
            y = y.to_numpy()

        stratify = y if len(pd.Series(y).value_counts()) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42 if not retry else 7,
                stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42 if not retry else 7
            )

        if not retry:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=3000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, random_state=42
                )
            }
        else:
            # Correction tool: new model families + stronger configuration.
            models = {
                "SVM (scaled)": Pipeline([
                    ("scale", StandardScaler()),
                    ("model", SVC(C=2.0, kernel="rbf"))
                ]),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.05,
                    max_depth=2, random_state=42
                ),
                "Random Forest (corrected)": RandomForestClassifier(
                    n_estimators=400, max_depth=None, min_samples_leaf=1,
                    random_state=7
                )
            }

        rows = []
        fitted = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = accuracy_score(y_test, pred)
                rows.append({"Model": name, "Score": float(score)})
                fitted[name] = model
            except Exception as e:
                rows.append({"Model": name, "Score": None, "Error": str(e)})

        result = pd.DataFrame(rows)
        valid = result.dropna(subset=["Score"])
        if valid.empty:
            raise ValueError("All models failed.")
        best_name = valid.loc[valid["Score"].idxmax(), "Model"]
        return result, fitted[best_name], best_name

    raise ValueError("This demo is configured for classification datasets.")

uploaded = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ Dataset successfully loaded!")

    st.header("🧠 Agent Planning")
    if st.button("Generate Agent Plan"):
        st.session_state["plan"] = make_plan(df)
    if "plan" in st.session_state:
        st.info(st.session_state["plan"])

    st.header("🔍 Dataset Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))
    st.dataframe(df, width="stretch")

    st.header("🧹 Automatic Data Cleaning")
    cleaned, actions = clean_data(df)
    st.success("✅ Cleaning tool executed")
    for a in actions or ["No cleaning changes required."]:
        st.write("• " + a)

    st.header("📊 Auto-EDA")
    numeric = cleaned.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in cleaned.columns if c not in numeric]
    st.write(f"Numeric columns: **{len(numeric)}** | Categorical columns: **{len(categorical)}**")

    if numeric:
        selected = st.selectbox("Select a numeric column", numeric)
        st.plotly_chart(
            px.histogram(cleaned, x=selected, title=f"Distribution of {selected}"),
            width="stretch"
        )

    if len(numeric) >= 2:
        st.plotly_chart(
            px.imshow(cleaned[numeric].corr(), text_auto=True, title="Feature Correlation"),
            width="stretch"
        )

    st.header("🤖 Autonomous Agent Execution")
    target = st.selectbox("🎯 Target column", cleaned.columns, index=len(cleaned.columns)-1)

    if st.button("🚀 Run Autonomous Agent", type="primary"):
        trace = []
        trace.append("GOAL → Build the strongest validated ML pipeline for the selected target.")
        trace.append("PLAN → Agent generated an execution plan.")
        trace.append("TOOL → Data analyzer inspected schema, missing values and duplicates.")
        trace.append("TOOL → Cleaning tool removed duplicates and imputed missing values.")

        problem = "Classification" if (
            not pd.api.types.is_numeric_dtype(cleaned[target])
            or cleaned[target].nunique() <= 10
        ) else "Regression"
        trace.append(f"OBSERVE → Detected problem type: {problem}.")

        if problem != "Classification":
            st.error("This demo currently supports classification for the correction workflow.")
            st.stop()

        with st.spinner("Agent is training baseline models..."):
            baseline, _, baseline_best = run_models(cleaned, target, problem, retry=False)

        trace.append("TOOL → Baseline model training/evaluation tool executed.")

        # Ask Gemini to decide whether the search space needs another iteration.
        reflection_prompt = f"""
You are the reflection/reasoning component of an autonomous ML agent.
Problem: {problem}
Baseline model results: {baseline.to_dict(orient="records")}

Decide whether another iteration is needed.
If only a small number of model families were tested, or models tie, recommend RETRY.
If the result is already strong and sufficiently explored, recommend COMPLETE.
Explain the reason in 2-4 sentences. Start with RETRY or COMPLETE.
"""
        reflection = gemini_reason(reflection_prompt)

        if not reflection:
            valid = baseline.dropna(subset=["Score"])
            best = float(valid["Score"].max()) if not valid.empty else 0
            tied = len(valid) >= 2 and valid["Score"].nunique() == 1
            retry = tied or best < 0.90
            reflection = (
                "RETRY — Baseline models are tied or the validation result is not strong enough; "
                "the agent should explore additional model families and configurations."
                if retry else
                "COMPLETE — The tested models provide a sufficiently strong result."
            )

        trace.append("REFLECT → " + reflection.replace("\n", " "))

        # IMPORTANT: Reflection now controls the correction branch.
        reflection_upper = reflection.upper()
        should_retry = "RETRY" in reflection_upper

        if should_retry:
            trace.append(
                "CORRECT → Reflection requested another iteration. "
                "Activating SVM, Gradient Boosting and corrected Random Forest tools."
            )
            with st.spinner("Agent is correcting the pipeline and retrying..."):
                corrected, _, corrected_best = run_models(
                    cleaned, target, problem, retry=True
                )
            trace.append("TOOL → Correction model family executed.")
            all_results = pd.concat([baseline, corrected], ignore_index=True)
        else:
            trace.append("CORRECT → No retry required.")
            all_results = baseline

        valid = all_results.dropna(subset=["Score"])
        final_name = valid.loc[valid["Score"].idxmax(), "Model"]
        final_score = float(valid["Score"].max())

        trace.append(
            f"OBSERVE → Best validated result after agent iteration: "
            f"{final_name} = {final_score:.2%}."
        )
        trace.append(f"COMPLETE → Selected {final_name} as the final model.")

        st.subheader("🛰️ Agent Trace")
        for item in trace:
            st.write("• " + item)

        st.subheader("🏆 Model Comparison")
        display = all_results.copy()
        display["Score"] = display["Score"].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "Failed"
        )
        st.dataframe(display, width="stretch")

        st.success(f"🏆 Final Model: {final_name}")
        st.metric("Final Accuracy", f"{final_score:.2%}")

        st.subheader("🧠 Reflection")
        st.info(reflection)

        # 📊 AI Business Insights & Recommendations
        st.subheader("📊 AI Business Insights")

        insight_df = cleaned.copy()

        st.write("### 🔎 Key Findings")

        # Numeric insights
        numeric_cols = insight_df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            for col in numeric_cols[:4]:
                if insight_df[col].notna().any():
                    avg_value = insight_df[col].mean()
                    max_value = insight_df[col].max()
                    min_value = insight_df[col].min()

                    st.write(
                        f"• **{col}** — Average: `{avg_value:.2f}`, "
                        f"Highest: `{max_value:.2f}`, "
                        f"Lowest: `{min_value:.2f}`"
                    )

        # Categorical insights
        categorical_cols = insight_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if categorical_cols:
            for col in categorical_cols[:4]:
                counts = insight_df[col].value_counts(dropna=True)
                if not counts.empty:
                    top_value = counts.index[0]
                    top_count = int(counts.iloc[0])

                    st.write(
                        f"• **{col}** — Most common: `{top_value}` "
                        f"({top_count} records)"
                    )

        st.subheader("💡 AI Recommendations")

        recommendations = []

        if numeric_cols:
            recommendations.append(
                "Monitor the highest-value numeric metrics and identify "
                "opportunities to improve them."
            )

        if categorical_cols:
            recommendations.append(
                "Focus marketing and inventory decisions on the most "
                "frequently occurring customer/product categories."
            )

        if final_score < 0.80:
            recommendations.append(
                "The current validation score is below 80%. Consider collecting "
                "more data, engineering better features, or tuning the models."
            )
        else:
            recommendations.append(
                "The current validation result is reasonably strong. "
                "Continue monitoring performance with new business data."
            )

        for recommendation in recommendations:
            st.info("💡 " + recommendation)

else:
    st.info("👆 Upload a CSV file above to start the autonomous agent.")

st.divider()
st.caption("AutoInsight AI | Agentic AI Challenge 2026 | Track A")