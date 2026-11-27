
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
    classifier = joblib.load("wealth_quintile_model.pkl")
    regressor = joblib.load("wealth_score_model.pkl")
    metadata = joblib.load("wealth_app_metadata.pkl")
    label_encoder = joblib.load("wealth_label_encoder.pkl")
    return classifier, regressor, metadata, label_encoder

classifier, regressor, metadata, label_encoder = load_models()

st.title("🏠 Abbay Basin Wealth Predictor")
st.write(
    "Machine Learning Based Household Wealth "
    "Prediction System - Abbay Basin, Ethiopia."
)

st.divider()
st.subheader("Household Characteristics")

feature_labels = {
    "Elevationm": "Elevation (meters)",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "hv009": "Household Members",
    "hv201": "Drinking Water Source",
    "hv205": "Toilet Facility",
    "hv206": "Electricity Availability",
    "hv207": "Radio Availability",
    "hv208": "Television Availability",
    "hv025": "Place of Residence",
}

water_labels = {
    10: "Piped water",
    11: "Piped into dwelling",
    12: "Piped to yard/plot",
    13: "Piped to neighbor",
    14: "Public tap/standpipe",
    20: "Tube well water",
    21: "Tube well or borehole",
    30: "Dug well (open/protected)",
    31: "Protected well",
    32: "Unprotected well",
    40: "Surface from spring",
    41: "Protected spring",
    42: "Unprotected spring",
    43: "River/dam/lake/pond/stream/canal/irrigation channel",
    51: "Rainwater",
    61: "Tanker truck",
    62: "Cart with small tank",
    71: "Bottled water",
    72: "Large bottle",
    96: "Other",
}

toilet_labels = {
    10: "Flush toilet",
    11: "Flush to piped sewer system",
    12: "Flush to septic tank",
    13: "Flush to pit latrine",
    14: "Flush to somewhere else",
    15: "Flush, don't know where",
    20: "Pit toilet latrine",
    21: "Ventilated improved pit latrine (VIP)",
    22: "Pit latrine with slab",
    23: "Pit latrine without slab/open pit",
    30: "No facility",
    31: "No facility/bush/field",
    41: "Composting toilet",
    42: "Bucket toilet",
    43: "Hanging toilet/latrine",
    96: "Other",
}

yes_no_labels = {0: "No", 1: "Yes"}
residence_labels = {1: "Urban", 2: "Rural"}

display_options = {
    "hv201": water_labels,
    "hv205": toilet_labels,
    "hv206": yes_no_labels,
    "hv207": yes_no_labels,
    "hv208": yes_no_labels,
    "hv025": residence_labels,
}

user_inputs = {}

for feature in metadata["features"]:

    # Hide hv012 from the application
    # but keep its default value for the trained model.
    if feature == "hv012":
        user_inputs[feature] = metadata["numeric_defaults"][feature]
        continue

    label = feature_labels.get(feature, feature)

    if feature in metadata["categorical_features"]:
        options = metadata["category_options"][feature]

        if feature in display_options:
            labels = display_options[feature]
            selected = st.selectbox(
                label,
                options=options,
                format_func=lambda value, labels=labels: labels.get(
                    int(value), f"Other (Code {value})"
                ),
                key=feature,
            )
        else:
            selected = st.selectbox(label, options=options, key=feature)

        user_inputs[feature] = selected

    else:
        minimum = metadata["numeric_min"][feature]
        maximum = metadata["numeric_max"][feature]
        default = metadata["numeric_defaults"][feature]

        if minimum == maximum:
            minimum -= 1
            maximum += 1

        selected = st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
            key=feature,
        )
        user_inputs[feature] = selected

st.divider()

if st.button(
    "Predict Household Wealth",
    type="primary",
    use_container_width=True,
):
    input_df = pd.DataFrame(
        [user_inputs],
        columns=metadata["features"],
    )

    predicted_class = classifier.predict(input_df)[0]
    wealth_category = label_encoder.inverse_transform(
        [predicted_class]
    )[0]
    predicted_score = regressor.predict(input_df)[0]

    st.success("Prediction Completed!")
    st.subheader("Prediction Results")

    st.metric(
        "Predicted Wealth Index Score (hv271)",
        f"{predicted_score:,.2f}",
    )

    st.metric(
        "Estimated Wealth Quintile",
        str(wealth_category).capitalize(),
    )

    st.info(
        "This is a machine learning estimate, "
        "not an official DHS wealth classification. "
        "Predictions may be less reliable for "
        "households outside the training data."
    )

st.divider()
st.caption("Senior Data Science Project | Abbay Basin, Ethiopia")
