import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Ride Cancellation Prediction",
    page_icon="🚕",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "ride_cancellation_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🚕 Ride Cancellation Prediction")
st.caption("Machine Learning & Spatio-Temporal Modeling")
st.write("Academic project | Concepts in Machine Learning (PBCMT304)")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset Analysis", "Model Evaluation",
     "Predict Cancellation", "About Project"]
)

if page == "Home":
    st.header("Welcome")
    st.write(
        "This project uses machine learning to estimate whether "
        "a ride booking may be cancelled."
    )
    st.subheader("Project objectives")
    st.markdown("""
    - Study ride-booking and cancellation patterns.
    - Prepare booking data for machine learning.
    - Train a Random Forest classification model.
    - Evaluate performance using classification metrics.
    - Demonstrate cancellation prediction through a website.
    """)
    st.info(
        "The model was trained on a dataset of 150,000 ride bookings. "
        "Predictions are estimates, not guarantees."
    )

elif page == "Dataset Analysis":
    st.header("Dataset Analysis")
    st.write("Dataset size: 150,000 bookings and 21 original columns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total bookings", "150,000")
    c2.metric("Cancelled", "37,500")
    c3.metric("Cancellation rate", "25.0%")

    status_data = pd.DataFrame({
        "Booking Status": [
            "Completed",
            "Cancelled by Driver",
            "No Driver Found",
            "Cancelled by Customer",
            "Incomplete"
        ],
        "Bookings": [93000, 27000, 10500, 10500, 9000]
    })

    st.subheader("Booking status distribution")
    st.bar_chart(status_data.set_index("Booking Status"))
    st.dataframe(status_data, use_container_width=True)

    st.caption(
        "For this model, statuses containing 'Cancelled' are labelled "
        "as cancellations. 'No Driver Found' and 'Incomplete' are not "
        "labelled as cancellations."
    )

elif page == "Model Evaluation":
    st.header("Random Forest Model Evaluation")
    st.write("Results from the 20% held-out test set (30,000 bookings).")

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "92.94%")
    c2.metric("Precision — cancellation", "77.98%")

    c3, c4 = st.columns(2)
    c3.metric("Recall — cancellation", "100.00%")
    c4.metric("F1-score — cancellation", "87.63%")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [0.9294, 0.7798, 1.0000, 0.8763]
    })
    st.dataframe(metrics, use_container_width=True)

    st.warning(
        "The test split was random. Perfect recall on this test set does "
        "not guarantee perfect detection on future bookings. The model "
        "may also flag some bookings that would not be cancelled."
    )

elif page == "Predict Cancellation":
    st.header("Predict a Ride Cancellation")
    st.write("Enter the booking details below.")

    try:
        model = load_model()

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                hour = st.slider("Booking hour", 0, 23, 12)
                day = st.selectbox(
                    "Day of week",
                    options=list(range(7)),
                    format_func=lambda x: [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ][x]
                )
                month = st.slider("Month", 1, 12, 6)
                vehicle = st.selectbox(
                    "Vehicle type",
                    ["Auto", "Bike", "eBike", "Go Sedan",
                     "Premier Sedan", "SUV", "Other"]
                )
                payment = st.selectbox(
                    "Payment method",
                    ["UPI", "Cash", "Debit Card", "Credit Card",
                     "Other", "Unknown"]
                )

            with col2:
                pickup = st.text_input(
                    "Pickup location", "Palam Vihar"
                )
                drop = st.text_input(
                    "Drop location", "Jhilmil"
                )
                distance = st.number_input(
                    "Ride distance (km)", min_value=0.0,
                    max_value=300.0, value=10.0, step=1.0
                )
                value = st.number_input(
                    "Booking value", min_value=0.0,
                    max_value=100000.0, value=300.0, step=50.0
                )

            submitted = st.form_submit_button(
                "Predict Cancellation", use_container_width=True
            )

        if submitted:
            if not pickup.strip() or not drop.strip():
                st.error("Please enter both pickup and drop locations.")
            else:
                input_data = pd.DataFrame([{
                    "hour": hour,
                    "day_of_week": day,
                    "month": month,
                    "Vehicle Type": vehicle,
                    "Pickup Location": pickup.strip(),
                    "Drop Location": drop.strip(),
                    "Ride Distance": distance,
                    "Booking Value": value,
                    "Payment Method": payment
                }])

                prediction = int(model.predict(input_data)[0])
                probability = float(
                    model.predict_proba(input_data)[0][1]
                )

                st.subheader("Prediction result")
                st.metric(
                    "Estimated cancellation probability",
                    f"{probability:.1%}"
                )
                st.progress(probability)

                if prediction == 1:
                    st.warning(
                        "The model predicts that this booking may be cancelled."
                    )
                else:
                    st.success(
                        "The model predicts that this booking may not be cancelled."
                    )

                st.caption(
                    "This is a machine-learning estimate, not a guarantee. "
                    "Unfamiliar locations or vehicle types may affect reliability."
                )

    except FileNotFoundError:
        st.error(
            "Model file not found. Put ride_cancellation_model.joblib "
            "in the same folder as app.py."
        )
    except Exception as error:
        st.error(f"Could not run prediction: {error}")

elif page == "About Project":
    st.header("About the Project")
    st.write(
        "This academic project explores machine learning for ride "
        "cancellation prediction using booking time, locations, vehicle "
        "type, distance, booking value and payment method."
    )
    st.subheader("Technology used")
    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - Streamlit
    - Google Colab
    """)
    st.subheader("PBCMT304 concepts demonstrated")
    st.markdown("""
    - Data preparation and feature selection
    - Classification using Random Forest
    - Training and test datasets
    - Accuracy, precision, recall and F1-score
    - Model deployment through a web application
    """)
    st.caption(
        "Project results are based on the training run reported by the team."
    )
