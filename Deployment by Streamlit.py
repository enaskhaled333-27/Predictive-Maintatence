import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT = joblib.load(BASE_DIR / "model.pkl")

model = ARTIFACT["model"]
scaler = ARTIFACT["scaler"]
feature_order = ARTIFACT["feature_order"]
numerical_columns = ARTIFACT["numerical_columns"]
type_categories = ARTIFACT["type_categories"]


def preprocess_input(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """Reproduce the notebook's inference preprocessing exactly."""
    row = pd.DataFrame(
        [
            {
                "Type": product_type,
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }
        ]
    )

    encoded = pd.DataFrame(
        {
            f"Type_{category}": (row["Type"] == category).astype(int)
            for category in type_categories
        },
        index=row.index,
    )

    numeric_part = row.drop(columns=["Type"])
    encoded = pd.concat([encoded, numeric_part], axis=1)[feature_order]

    encoded[numerical_columns] = scaler.transform(encoded[numerical_columns])
    return encoded


st.markdown(
    """
    <style>
        .main {
            background: #f7f9fc;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, #0f4c81 0%, #2563eb 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.15);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            font-weight: 750;
        }

        .hero p {
            margin: 0.55rem 0 0;
            font-size: 1.05rem;
            opacity: 0.92;
        }

        .section-card {
            background: white;
            border: 1px solid #e5eaf2;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #16324f;
            margin-bottom: 0.7rem;
        }

        .result-card {
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
            border: 1px solid #e5eaf2;
        }

        .result-failure {
            background: #fff5f5;
            border-color: #fecaca;
        }

        .result-safe {
            background: #f0fdf4;
            border-color: #bbf7d0;
        }

        .interpretation {
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 1rem 1.15rem;
            border-radius: 8px;
            color: #334155;
            line-height: 1.55;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            min-height: 3rem;
            font-weight: 700;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Predictive Maintenance - Machine Failure Prediction</h1>
        <p>AI-powered machine failure prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Machine Operating Conditions</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    product_type = st.selectbox(
        "Product Type",
        options=["L", "M", "H"],
        index=0,
        help="Machine product quality variant.",
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1,
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=310.0,
        step=0.1,
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=4000,
        value=1500,
        step=1,
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1,
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1,
    )

st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict Failure", type="primary", use_container_width=True):
    try:
        features = preprocess_input(
            product_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )

        prediction = int(model.predict(features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0, 1])

        if prediction == 1:
            st.markdown(
                """
                <div class="result-card result-failure">
                    <h2 style="color:#b91c1c; margin:0;">⚠️ Machine Failure Predicted</h2>
                    <p style="margin:0.5rem 0 0; color:#7f1d1d;">
                        The model predicts that the entered operating conditions are associated with a machine failure.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card result-safe">
                    <h2 style="color:#15803d; margin:0;">✅ No Machine Failure Predicted</h2>
                    <p style="margin:0.5rem 0 0; color:#166534;">
                        The model does not predict a machine failure for the entered operating conditions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if probability is not None:
            st.metric("Failure Probability", f"{probability:.1%}")

    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediction Interpretation</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="interpretation">
        This prediction is based on the machine operating conditions entered above:
        product type, air temperature, process temperature, rotational speed, torque,
        and tool wear. The application applies the same categorical encoding and
        numerical scaling used during training before passing the data to the final
        Decision Tree model.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.caption("Deployment model: Decision Tree Classifier • Target: Machine Failure (0/1)")
