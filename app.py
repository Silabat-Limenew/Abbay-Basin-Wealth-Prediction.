
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Abbay Basin Wealth Predictor",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def load_models():

    classifier = joblib.load(
        'wealth_quintile_model.pkl'
    )

    regressor = joblib.load(
        'wealth_score_model.pkl'
    )

    metadata = joblib.load(
        'wealth_app_metadata.pkl'
    )

    return classifier, regressor, metadata


classifier, regressor, metadata = load_models()

st.title("🏠 Abbay Basin Wealth Predictor")

st.write(
    "Machine Learning Based Household Wealth "
    "Prediction System - Abbay Basin, Ethiopia."
)

st.divider()

st.subheader("Household Characteristics")

feature_labels = {
    'Elevationm': 'Elevation (meters)',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'hv009': 'Household Members',
    'hv012': 'De Jure Household Members',
    'hv201': 'Drinking Water Source',
    'hv205': 'Toilet Facility',
    'hv206': 'Electricity Availability',
    'hv207': 'Radio Availability',
    'hv208': 'Television Availability',
    'hv025': 'Place of Residence'
}

user_inputs = {}

for feature in metadata['features']:

    label = feature_labels.get(feature, feature)

    if feature in metadata['categorical_features']:

        options = metadata['category_options'][feature]

        selected = st.selectbox(
            label,
            options=options,
            key=feature
        )

        user_inputs[feature] = selected

    else:

        minimum = metadata['numeric_min'][feature]

        maximum = metadata['numeric_max'][feature]

        default = metadata['numeric_defaults'][feature]

        if minimum == maximum:
            minimum -= 1
            maximum += 1

        selected = st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            key=feature
        )

        user_inputs[feature] = selected

st.divider()

if st.button(
    "Predict Household Wealth",
    type="primary",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [user_inputs],
        columns=metadata['features']
    )

    # Classification prediction
    predicted_class = classifier.predict(
        input_df
    )[0]

    label_encoder = joblib.load(
        'wealth_label_encoder.pkl'
    )

    wealth_category = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # Regression prediction
    predicted_score = regressor.predict(
        input_df
    )[0]

    st.success("Prediction Completed!")

    st.subheader("Prediction Results")

    st.metric(
        "Estimated Wealth Quintile",
        wealth_category.capitalize()
    )

    st.metric(
        "Predicted Wealth Index Score (hv271)",
        f"{predicted_score:,.2f}"
    )

    st.info(
        "This is a machine learning estimate, "
        "not an official DHS wealth classification. "
        "Predictions may be less reliable for "
        "households outside the training data."
    )

st.divider()

st.caption(
    "Senior Data Science Project | "
    "Abbay Basin, Ethiopia"
)
