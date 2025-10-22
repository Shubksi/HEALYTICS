# ============================================
#  HEALYTICS – CLAIM DENIAL PREDICTOR + LLM
#  CatBoost + Gemma3:1b (via Ollama)
# ============================================

import os
import json
import requests
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier, Pool

# ✅ Streamlit configuration (must come first)
st.set_page_config(page_title="Claim Denial Predictor", page_icon="🩺", layout="centered")

# ============================================
# 1️⃣ PATHS & FEATURE SETUP
# ============================================
MODEL_PATH = Path("models/catboost_model.cbm")
ENCODER_PATH = Path("models/label_encoder.pkl")

categorical_cols = [
    "Gender", "Payer", "Plan_Type", "CPT", "ICD10_Codes",
    "Dx_Pairing_Validity", "Modifier", "Provider_Specialty",
    "Place_of_Service", "Network_Status", "Auth_Status",
    "Documentation_Status"
]
numeric_cols = [
    "Age", "Eligibility", "Coverage_Active", "Submission_Lag_Days", "TFL",
    "Charge_Amount", "Benefit_Usage_Perc", "COB_Primary_Billed",
    "Experimental_Flag", "Member_ID_Valid"
]
features = categorical_cols + numeric_cols
cat_features_idx = [features.index(c) for c in categorical_cols]

# ============================================
# 2️⃣ LOAD MODEL + ENCODER
# ============================================
@st.cache_resource(show_spinner="Loading CatBoost model and label encoder…")
def load_model_and_encoder():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing label encoder: {ENCODER_PATH}")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder

try:
    model, label_encoder = load_model_and_encoder()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# ============================================
# 3️⃣ HEADER + SIDEBAR
# ============================================
st.title("🩺 Healthcare Claim Denial Predictor")
st.caption("Powered by CatBoost + Local Gemma3:1b (via Ollama)")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "Predicts the **Primary Denial Reason** using a trained CatBoost model, "
        "and optionally generates a short natural-language explanation using **Gemma3:1b** running locally through Ollama."
    )
    st.markdown("**Artifacts:**")
    st.code(f"{MODEL_PATH}\n{ENCODER_PATH}", language="bash")
    st.markdown("**Tip:** Make sure Ollama is running (`ollama serve`) and the model `gemma3:1b` is pulled.")
    st.divider()

# ============================================
# 4️⃣ INPUT FORM
# ============================================
with st.form("claim_form"):
    st.subheader("Enter Claim Details")

    col1, col2 = st.columns(2)
    Gender = col1.selectbox("Gender", ["M", "F", "U"])
    Payer = col2.text_input("Payer", "Aetna")
    Plan_Type = col1.selectbox("Plan Type", ["HMO", "PPO", "EPO", "POS", "Other"], index=1)
    CPT = col2.text_input("CPT", "99213")
    ICD10_Codes = col1.text_input("ICD-10 Codes", "E11.9")
    Dx_Pairing_Validity = col2.selectbox("DX Pairing Validity", ["Valid", "Invalid", "Unknown"], index=0)
    Modifier = col1.text_input("Modifier", "25")
    Provider_Specialty = col2.text_input("Provider Specialty", "Internal_Medicine")
    Place_of_Service = col1.text_input("Place of Service", "11")
    Network_Status = col2.selectbox("Network Status", ["In-Network", "Out-of-Network"], index=0)
    Auth_Status = col1.selectbox("Authorization Status", ["Approved", "Required", "Not-Required", "Denied"], index=0)
    Documentation_Status = col2.selectbox("Documentation Status", ["Complete", "Incomplete", "Missing"], index=0)

    Age = st.number_input("Age", min_value=0, max_value=120, value=47)
    Eligibility = st.number_input("Eligibility (0/1)", min_value=0, max_value=1, value=1)
    Coverage_Active = st.number_input("Coverage Active (0/1)", min_value=0, max_value=1, value=1)
    Submission_Lag_Days = st.number_input("Submission Lag (days)", min_value=0, max_value=365, value=5)
    TFL = st.number_input("Timely Filing Limit (days)", min_value=0, max_value=3650, value=365)
    Charge_Amount = st.number_input("Charge Amount", min_value=0.0, value=145.0, step=1.0, format="%.2f")
    Benefit_Usage_Perc = st.number_input("Benefit Usage %", min_value=0.0, max_value=100.0, value=32.5, step=0.1)
    COB_Primary_Billed = st.number_input("COB Primary Billed (0/1)", min_value=0, max_value=1, value=0)
    Experimental_Flag = st.number_input("Experimental Flag (0/1)", min_value=0, max_value=1, value=0)
    Member_ID_Valid = st.number_input("Member ID Valid (0/1)", min_value=0, max_value=1, value=1)

    use_llm = st.checkbox("🧠 Generate Explanation with Gemma3:1b")
    submitted = st.form_submit_button("Predict")

# ============================================
# 5️⃣ HELPER FUNCTIONS
# ============================================
def make_claim_df():
    row = {
        "Gender": Gender, "Payer": Payer, "Plan_Type": Plan_Type, "CPT": CPT, "ICD10_Codes": ICD10_Codes,
        "Dx_Pairing_Validity": Dx_Pairing_Validity, "Modifier": Modifier, "Provider_Specialty": Provider_Specialty,
        "Place_of_Service": Place_of_Service, "Network_Status": Network_Status, "Auth_Status": Auth_Status,
        "Documentation_Status": Documentation_Status, "Age": Age, "Eligibility": Eligibility,
        "Coverage_Active": Coverage_Active, "Submission_Lag_Days": Submission_Lag_Days, "TFL": TFL,
        "Charge_Amount": Charge_Amount, "Benefit_Usage_Perc": Benefit_Usage_Perc,
        "COB_Primary_Billed": COB_Primary_Billed, "Experimental_Flag": Experimental_Flag,
        "Member_ID_Valid": Member_ID_Valid
    }
    df = pd.DataFrame([row], columns=features)
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("NA")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def generate_ollama_explanation(claim_data, prediction):
    """
    Generate local explanation via Ollama + Gemma3:1b
    """
    prompt = f"""
    The following healthcare claim was analyzed:
    {json.dumps(claim_data, indent=2)}
    The model predicted the denial reason: "{prediction}".
    Explain briefly why such a claim might be denied based on typical medical billing or insurance rules.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt},
            stream=True,
            timeout=120
        )
        text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                text += data.get("response", "")
        return text.strip() or "No response from Gemma3."
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

# ============================================
# 6️⃣ PREDICTION + EXPLANATION
# ============================================
if submitted:
    try:
        df_raw = make_claim_df()
        pool = Pool(df_raw, cat_features=cat_features_idx)
        probs = model.predict_proba(pool)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.success(f"🩺 **Predicted Denial Reason:** {pred_label}")

        st.subheader("Top Prediction Probabilities")
        top = sorted(
            [(label_encoder.inverse_transform([i])[0], float(p)) for i, p in enumerate(probs)],
            key=lambda kv: kv[1],
            reverse=True
        )[:5]
        st.table(pd.DataFrame(top, columns=["Denial Reason", "Probability"])
                 .assign(Probability=lambda d: d["Probability"].round(3)))

        with st.expander("Preview of Model Input (Debugging)"):
            st.dataframe(df_raw)

        # --- Local Gemma3 explanation ---
        if use_llm:
            st.subheader("🧠 Local LLM Explanation (Gemma3:1b)")
            with st.spinner("Generating explanation with Gemma3:1b..."):
                claim_dict = df_raw.to_dict(orient="records")[0]
                explanation = generate_ollama_explanation(claim_dict, pred_label)
            st.info(explanation)

    except Exception as e:
        st.error(f"Inference error: {e}")


# ============================================
# 7️⃣ CPT–ICD COMPATIBILITY TOOL (Expandable)
# ============================================

import os
import json
import requests
import pandas as pd
import streamlit as st

st.markdown("---")
st.header("🧰 CPT–ICD Compatibility Checker")

@st.cache_data
def load_cpt_icd_mapping(path="data/CPT-DX.csv"):
    """Load CPT–ICD mapping from simplified CSV with two columns."""
    if not os.path.exists(path):
        st.warning("⚠️ CPT–DX.csv not found in the 'data/' folder. You can still get AI explanations.")
        return {}

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]

    # Expect: 'CPT CODE' and 'ICD-10 CODES'
    if "CPT CODE" not in df.columns or "ICD-10 CODES" not in df.columns:
        st.warning(f"⚠️ CSV should have columns 'CPT Code' and 'ICD-10 Codes'. Found: {list(df.columns)}")
        return {}

    # Split multiple ICDs separated by commas or semicolons
    df["__ICD_LIST__"] = df["ICD-10 CODES"].fillna("").apply(
        lambda x: [i.strip().upper() for i in str(x).replace(";", ",").split(",") if i.strip()]
    )

    mapping = df.groupby("CPT CODE")["__ICD_LIST__"].sum()
    mapping = {str(k).strip().upper(): sorted(set(v)) for k, v in mapping.items()}
    return mapping


def check_cpt_icd_pair(mapping, cpt, icd):
    """Validate if CPT–ICD pair exists in mapping."""
    cpt = cpt.strip().upper()
    icd = icd.strip().upper()

    # Handle case when mapping file missing or empty
    if not mapping:
        return None, f"⚠️ No mapping file loaded. AI will still explain {cpt} and {icd}.", []

    if cpt not in mapping:
        return None, f"⚠️ CPT {cpt} not found in mapping. AI will still analyze compatibility.", []
    valid_icds = mapping[cpt]

    if icd in valid_icds:
        return True, f"✅ CPT {cpt} and ICD {icd} are valid.", valid_icds
    else:
        return False, f"❌ ICD {icd} not valid for CPT {cpt}. Valid ICDs: {', '.join(valid_icds)}", valid_icds


def explain_cpt_icd_with_gemma(cpt, icd, valid):
    """Explain CPT–ICD relationship using local Gemma (via Ollama HTTP API)."""
    if valid is True:
        validity_line = "This CPT–ICD pairing is clinically appropriate."
    elif valid is False:
        validity_line = "This CPT–ICD pairing may not be clinically appropriate, but may still be explainable."
    else:
        validity_line = "This CPT–ICD combination is not in the known mapping file, but we can still reason about it."

    prompt = f"""
    You are a medical billing assistant.
    Explain what CPT code {cpt} and ICD-10 code {icd} represent,
    and provide insight into whether they could be clinically related or justified.
    {validity_line}
    Keep the answer factual, concise (3–5 sentences), and in professional medical language.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt, "stream": False},
            timeout=180,
        )
        data = response.json()
        return data.get("response", "").strip() or "⚠️ No response from Gemma."
    except Exception as e:
        return f"⚠️ Gemma (Ollama) request error: {e}"


with st.expander("🔎 Check CPT–ICD Compatibility", expanded=False):
    mapping = load_cpt_icd_mapping()
    cpt_in = st.text_input("Enter CPT Code (e.g., 93306)", key="cpt_tool")
    icd_in = st.text_input("Enter ICD-10 Code (e.g., I50.9)", key="icd_tool")

    if st.button("Check Compatibility", key="check_btn"):
        valid, msg, valid_list = check_cpt_icd_pair(mapping, cpt_in, icd_in)

        # Always show message
        if valid is True:
            st.success(msg)
        elif valid is False:
            st.error(msg)
        else:
            st.info(msg)

        # Always generate AI explanation (even if not found)
        if cpt_in and icd_in:
            with st.spinner("💬 Gemma is preparing an explanation..."):
                explanation = explain_cpt_icd_with_gemma(cpt_in, icd_in, valid)
                st.info(explanation)

# ============================================
# 8️⃣ FLOATING CHATBOT – HEALYTICS ASSISTANT
# ============================================

import streamlit.components.v1 as components

st.markdown("---")

# Inject a floating button with HTML + CSS
chatbot_button = """
<style>
.chat-button {
  position: fixed;
  bottom: 30px;
  right: 30px;
  background-color: #2e86de;
  color: white;
  border: none;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  font-size: 28px;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  z-index: 9999;
}
.chat-window {
  position: fixed;
  bottom: 100px;
  right: 30px;
  width: 350px;
  height: 450px;
  background-color: #f8f9fa;
  border-radius: 10px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  display: none;
  flex-direction: column;
  overflow: hidden;
  z-index: 9998;
}
.chat-header {
  background-color: #2e86de;
  color: white;
  text-align: center;
  padding: 10px;
  font-weight: bold;
}
</style>

<button class="chat-button" onclick="toggleChat()">💬</button>

<div id="chatbox" class="chat-window">
  <div class="chat-header">HEALYTICS Assistant</div>
  <iframe srcdoc="<p style='padding:20px'>Loading chatbot...</p>" id="chatframe" style="border:none;width:100%;height:100%;"></iframe>
</div>

<script>
function toggleChat() {
  var chat = document.getElementById('chatbox');
  if (chat.style.display === 'none' || chat.style.display === '') {
    chat.style.display = 'flex';
    window.parent.postMessage({type: 'open_chat'}, '*');
  } else {
    chat.style.display = 'none';
  }
}
</script>
"""

components.html(chatbot_button, height=0, width=0)

# Handle internal chatbot logic
st.session_state.setdefault("chat_history", [])

def query_gemma(prompt):
    """Talk to local Gemma3:1b model via Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:1b", "prompt": prompt},
            stream=True,
            timeout=180,
        )
        text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                text += data.get("response", "")
        return text.strip()
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

# Render the actual chatbot UI (hidden inside iframe)
with st.sidebar.expander("💬 Chatbot Engine (hidden UI)", expanded=False):
    st.write("Internal Chat Engine – you can hide this section.")

    user_input = st.text_input("Ask HEALYTICS Assistant:", key="chat_input")
    if st.button("Send", key="chat_send"):
        if user_input:
            mapping = None
            # Load CPT-DX file if available
            path = "data/CPT-DX.csv"
            if os.path.exists(path):
                df = pd.read_csv(path, dtype=str)
                df.columns = [c.strip().upper() for c in df.columns]
                if "CPT CODE" in df.columns and "ICD-10 CODES" in df.columns:
                    mapping = df

            context = ""
            if mapping is not None:
                context = f"Reference CPT–ICD data sample:\n{mapping.head(10).to_json(orient='records')}"

            prompt = f"""
            You are HEALYTICS Assistant – a medical billing & claim denial support bot.
            User asked: "{user_input}"
            {context}
            Answer briefly and professionally.
            """

            answer = query_gemma(prompt)
            st.session_state.chat_history.append((user_input, answer))

    # Display conversation history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 HEALYTICS:** {a}")