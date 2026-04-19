import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Loan Approval System", layout="centered")

st.markdown("### 💡 Intelligent Loan Decision System")
st.markdown("---")

# ------------------ DATABASE ------------------
conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()

# Users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')

# Predictions table
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    username TEXT,
    income REAL,
    loan REAL,
    probability REAL,
    result TEXT
)
''')
conn.commit()

# ------------------ SESSION ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ------------------ AUTH ------------------
st.sidebar.title("🔐 Authentication")
auth_mode = st.sidebar.radio("Choose Option", ["Login", "Register"])

# REGISTER
if auth_mode == "Register":
    st.subheader("📝 Create Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Register"):
        if new_user and new_pass:
            try:
                c.execute("INSERT INTO users VALUES (?, ?)", (new_user, new_pass))
                conn.commit()
                st.success("Account created ✅")
            except:
                st.error("User already exists ❌")
        else:
            st.warning("Fill all fields")

# LOGIN
elif auth_mode == "Login":
    st.subheader("🔑 Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (user, password)
        ).fetchone()

        if result:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

# LOGOUT
if st.session_state.logged_in:
    st.sidebar.success(f"👤 {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# BLOCK
if not st.session_state.logged_in:
    st.warning("🔒 Please login to continue")
    st.stop()

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# ------------------ MAIN UI ------------------
st.title("🏦 Loan Approval Prediction System")

# INPUTS (NEW DATASET)
dependents = st.number_input("Number of Dependents", min_value=0)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.selectbox("Self Employed", ["Yes", "No"])

income = st.number_input("Annual Income")
loan = st.number_input("Loan Amount")
term = st.number_input("Loan Term")

cibil = st.slider("CIBIL Score", 300, 900, 650)

res_assets = st.number_input("Residential Assets Value")
com_assets = st.number_input("Commercial Assets Value")
lux_assets = st.number_input("Luxury Assets Value")
bank_assets = st.number_input("Bank Asset Value")

# ENCODING
education = 1 if education == "Graduate" else 0
self_emp = 1 if self_emp == "Yes" else 0

# INPUT ARRAY
input_data = np.array([[
    dependents, education, self_emp,
    income, loan, term, cibil,
    res_assets, com_assets, lux_assets, bank_assets
]])

# ------------------ PREDICTION ------------------
if st.button("Predict Loan Status"):

    prob = model.predict_proba(input_data)[0][1]
    result = "Approved" if prob > 0.5 else "Rejected"

    st.subheader(f"📊 Approval Probability: {prob*100:.2f}%")

    # CIBIL BASED RISK
    if cibil < 500:
        st.error("❌ High Risk (Low CIBIL Score)")
    elif cibil < 700:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk (Good Credit Score)")

    # SAVE
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
              (st.session_state.username, income, loan, prob, result))
    conn.commit()

    # CHART
    data = pd.DataFrame({
        "Metric": ["Income", "Loan", "Assets"],
        "Value": [income, loan, res_assets + com_assets + lux_assets]
    })

    fig = px.bar(data, x="Metric", y="Value", title="Financial Overview")
    st.plotly_chart(fig)

    # EXPLANATION
    st.subheader("📌 Decision Explanation")

    if cibil < 600:
        st.write("- ❗ Low CIBIL score impacts approval")

    if loan > income:
        st.write("- ⚠️ Loan amount higher than income")

    if (res_assets + bank_assets) < loan:
        st.write("- ⚠️ Low asset backing")

    if prob > 0.7:
        st.write("- ✅ Strong financial profile")

    # SUGGESTIONS
    st.subheader("💡 Suggestions")

    if prob < 0.5:
        st.write("✔ Improve credit score")
        st.write("✔ Reduce loan amount")
        st.write("✔ Increase assets/income")
    else:
        st.write("✔ Good financial standing")

# ------------------ ADMIN DASHBOARD ------------------
if st.session_state.username == "admin":
    st.subheader("📊 Admin Dashboard")

    data = c.execute("SELECT * FROM predictions").fetchall()

    if data:
        df = pd.DataFrame(data, columns=["User","Income","Loan","Probability","Result"])
        st.dataframe(df)