import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"   

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter the patient details below to estimate diabetes probability.")

# --- Input form ---
gender = st.selectbox("Gender", ["female", "male"])
age = st.number_input("Age", min_value=1, max_value=120, value=45)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "no info", "current", "former", "ever", "not current"]
)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=12.0, value=5.8)
blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=120
)

if st.button("Predict Diabetes Risk"):
    payload = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        p_diab = result["probability_diabetes"]
        p_no_diab = result["probability_no_diabetes"]
        pred_label = "Diabetic" if result["prediction"] == 1 else "Non-Diabetic"

        st.subheader("🔎 Prediction Result")
        st.write(f"**Prediction:** {pred_label}")
        st.write(f"**Probability (Diabetes):** {p_diab:.4f}")
        st.write(f"**Probability (No Diabetes):** {p_no_diab:.4f}")

        # --- Emoji face / interpretation ---
        st.markdown("---")
        st.subheader("🧠 Interpretation")

        if p_diab > 0.6:
            st.markdown("### 😢 High risk of diabetes")
            st.write(
                "The model estimates a relatively **high probability** of diabetes. "
                "This is **not a medical diagnosis**, only a model prediction."
            )
        elif p_diab < 0.5:
            st.markdown("### 😄 Low risk of diabetes")
            st.write(
                "The model estimates a **low probability** of diabetes. "
                "Still, maintaining healthy habits is important."
            )
        else:
            st.markdown("### 😐 Borderline risk")
            st.write(
                "The model estimates a **borderline probability** of diabetes. "
                "Small changes in lifestyle and regular check-ups can be helpful."
            )


        st.markdown("---")
        st.subheader("💡 Simple (non-medical) suggestions")

        suggestions = []


        if bmi >= 30:
            suggestions.append("Try to gradually reduce BMI through balanced diet and regular activity.")
        elif bmi >= 25:
            suggestions.append("Keep an eye on BMI and consider light exercise and balanced meals.")

        if blood_glucose_level >= 140:
            suggestions.append("Blood glucose is on the higher side; limiting sugary drinks/foods may help.")
        elif blood_glucose_level >= 110:
            suggestions.append("Blood glucose is slightly elevated; monitoring sugar intake could be useful.")

        if HbA1c_level >= 6.5:
            suggestions.append("HbA1c level is high; regular medical follow-up is important.")
        elif HbA1c_level >= 5.7:
            suggestions.append("HbA1c is in a higher range; healthy lifestyle may reduce future risk.")

        if smoking_history in ["current", "ever", "not current", "former"]:
            suggestions.append("Avoiding or quitting smoking is generally beneficial for overall health.")

        if hypertension == 1:
            suggestions.append("Managing blood pressure (less salt, more activity) can support long-term health.")

        if heart_disease == 1:
            suggestions.append("Heart disease history makes regular medical check-ups extra important.")

        if not suggestions:
            st.write(
                "No specific risk signals stood out strongly from the inputs. "
                "General healthy lifestyle and regular check-ups are always recommended."
            )
        else:
            for s in suggestions:
                st.write(f"- {s}")

        st.info(
            "⚠️ This tool is for **educational purposes only** and is **not medical advice**. "
            "Always consult a healthcare professional for real medical decisions."
        )

    except Exception as e:
        st.error("❌ Could not reach FastAPI backend or prediction failed.")
        st.error(str(e))
