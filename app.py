import streamlit as st
import pandas as pd
import joblib

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    .risk-card {
        border-radius: 16px; padding: 28px 20px; text-align: center;
        border: 2px solid; margin-bottom: 12px;
    }
    .risk-card h2 {margin: 0 0 6px 0; font-size: 1.9rem;}
    .risk-card .big {font-size: 3.2rem; font-weight: 800; line-height: 1.1;}
    .risk-card .sub {opacity: 0.85; margin-top: 6px;}
    .low  {background: rgba(46,160,67,0.15);  border-color: #2ea043; color: #2ea043;}
    .med  {background: rgba(240,173,0,0.15);  border-color: #f0ad00; color: #f0ad00;}
    .high {background: rgba(229,57,53,0.15);  border-color: #e53935; color: #e53935;}
    .placeholder {
        border: 2px dashed rgba(128,128,128,0.5); border-radius: 16px;
        padding: 60px 20px; text-align: center; opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return (
        joblib.load("model.pkl"),
        joblib.load("encoders.pkl"),
        joblib.load("columns.pkl"),
    )


def gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#555"},
                "steps": [
                    {"range": [0, 30], "color": "#2ea043"},
                    {"range": [30, 60], "color": "#f0ad00"},
                    {"range": [60, 100], "color": "#e53935"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def main():
    model, encoders, columns = load_models()

    def choice(label, col):
        return st.selectbox(label, list(encoders[col].classes_))

    # ---------- Header ----------
    st.title("📉 Customer Churn Prediction System")
    st.caption("Predict whether a customer will leave, based on their profile, account and services.")
    st.markdown("---")

    left, right = st.columns([3, 2], gap="large")

    # ---------- LEFT: inputs ----------
    with left:
        st.subheader("📝 Customer Information")

        with st.container(border=True):
            st.markdown("##### 👤 Demographics")
            a, b, c = st.columns(3)
            with a:
                gender = choice("Gender", "gender")
                senior = st.selectbox("Senior citizen", ["No", "Yes"])
            with b:
                partner = choice("Has partner", "Partner")
                dependents = choice("Has dependents", "Dependents")
            with c:
                tenure = st.slider("Tenure (months)", 0, 72, 12)

        with st.container(border=True):
            st.markdown("##### 💳 Account Information")
            a, b = st.columns(2)
            with a:
                contract = choice("Contract type", "Contract")
                paperless = choice("Paperless billing", "PaperlessBilling")
                monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
            with b:
                payment = choice("Payment method", "PaymentMethod")
                total = st.number_input(
                    "Total charges ($)", 0.0, 10000.0, float(round(monthly * tenure, 2)), step=10.0
                )

        with st.container(border=True):
            st.markdown("##### 📡 Services")
            a, b, c = st.columns(3)
            with a:
                phone = choice("Phone service", "PhoneService")
                lines = choice("Multiple lines", "MultipleLines")
                internet = choice("Internet service", "InternetService")
            with b:
                security = choice("Online security", "OnlineSecurity")
                backup = choice("Online backup", "OnlineBackup")
                protection = choice("Device protection", "DeviceProtection")
            with c:
                support = choice("Tech support", "TechSupport")
                tv = choice("Streaming TV", "StreamingTV")
                movies = choice("Streaming movies", "StreamingMovies")

        predict = st.button("🔮 Predict churn", type="primary", use_container_width=True)

    # ---------- Prediction ----------
    if predict:
        row = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        df = pd.DataFrame([row])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        df = df[columns]
        st.session_state["prob"] = float(model.predict_proba(df)[0][1])
        st.session_state["summary"] = {
            "Contract": contract, "Tenure": f"{tenure} months",
            "Monthly charges": f"${monthly:,.2f}", "Internet": internet,
            "Payment": payment,
        }

    # ---------- RIGHT: results ----------
    with right:
        st.subheader("🎯 Prediction Results")
        prob = st.session_state.get("prob")

        if prob is None:
            st.markdown(
                '<div class="placeholder">Fill in the customer details<br>'
                'and click <b>Predict churn</b></div>',
                unsafe_allow_html=True,
            )
        else:
            if prob < 0.30:
                cls, icon, level, msg = "low", "✅", "LOW RISK", "Customer is likely to stay"
            elif prob < 0.60:
                cls, icon, level, msg = "med", "⚠️", "MEDIUM RISK", "Customer may leave, keep an eye on them"
            else:
                cls, icon, level, msg = "high", "🚨", "HIGH RISK", "Customer is likely to leave"

            st.markdown(
                f"""
                <div class="risk-card {cls}">
                    <h2>{icon} {level}</h2>
                    <div class="big">{prob:.1%}</div>
                    <div class="sub">{msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Churn probability", f"{prob:.1%}")
            m2.metric("Retention probability", f"{1 - prob:.1%}")

            if HAS_PLOTLY:
                st.plotly_chart(gauge(prob), use_container_width=True)
            else:
                st.progress(prob)

            if prob >= 0.30:
                st.info("💡 **Retention ideas:** offer a longer contract discount, "
                        "bundle tech support or online security, or move the customer "
                        "to automatic payment.")

            with st.expander("📋 Customer summary"):
                for k, v in st.session_state["summary"].items():
                    st.write(f"**{k}:** {v}")

            with st.expander("📊 What drives churn in this model?"):
                imp = (pd.Series(model.feature_importances_, index=columns)
                       .sort_values(ascending=False).head(8))
                st.bar_chart(imp)


if __name__ == "__main__":
    main()
