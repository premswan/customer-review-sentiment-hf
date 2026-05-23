import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("Customer Review Sentiment Analyzer")
st.write("Enter a customer review and classify it as Positive or Negative.")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

review = st.text_area(
    "Enter customer review:",
    placeholder="Example: The flight was on time and the staff were helpful."
)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = classifier(review)[0]

        label = result["label"]
        score = result["score"]

        st.subheader("Prediction Result")

        if label == "POSITIVE":
            st.success("Sentiment: Positive")
        else:
            st.error("Sentiment: Negative")

        st.write(f"Confidence Score: {score:.4f}")

        st.subheader("Business Interpretation")
        if label == "POSITIVE":
            st.write(
                "The customer feedback indicates satisfaction. "
                "This can help the business identify strong service areas."
            )
        else:
            st.write(
                "The customer feedback indicates dissatisfaction. "
                "This should be reviewed by the support or operations team."
            )
