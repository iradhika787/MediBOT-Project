import re
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, render_template, request
from fuzzywuzzy import fuzz, process

from medibot.services.ml_predictor import MLPredictionError, predict_symptoms


chat_bp = Blueprint("chat", __name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPOSITORY_ROOT / "data" / "medical_ds.csv"
FUZZY_MATCH_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "me",
    "my",
    "we",
    "you",
    "he",
    "she",
    "it",
    "what",
    "how",
    "when",
    "where",
    "why",
}
MISSING_TREATMENT_MESSAGE = (
    "Treatment information is currently unavailable. "
    "Please consult a qualified healthcare professional."
)


def _load_dataset():
    try:
        dataframe = pd.read_csv(DATASET_PATH)
        if "Disease" in dataframe.columns and "Name" not in dataframe.columns:
            dataframe["Name"] = dataframe["Disease"]
        dataframe.dropna(subset=["Symptoms", "Name"], inplace=True)
        dataframe["Symptoms"] = dataframe["Symptoms"].apply(
            lambda value: ",".join(
                sorted(
                    symptom.strip().lower()
                    for symptom in str(value).split(",")
                    if symptom.strip()
                )
            )
        )
        print(f"CSV loaded from {DATASET_PATH}.")
        return dataframe
    except Exception as error:
        print(f"Error loading dataset from {DATASET_PATH}: {error}")
        return pd.DataFrame(columns=["Name", "Symptoms", "Treatments"])


df = _load_dataset()


@chat_bp.get("/")
def home():
    return render_template("index.html")


@chat_bp.get("/features.html")
def features():
    return render_template("features.html")


@chat_bp.get("/chatbot.html")
def chatbot_page():
    return render_template("chatbot.html")


@chat_bp.get("/APP.HTML")
def appointment_page():
    return render_template("APP.HTML")


@chat_bp.get("/med.html")
def medicine_page():
    return render_template("med.html")


def fuzzy_match_symptom(input_symptom, all_symptoms):
    match = process.extractOne(input_symptom, all_symptoms, scorer=fuzz.ratio)
    if match and match[1] > 70:
        return match[0]
    return None


@chat_bp.post("/chat")
def chat():
    try:
        data = request.get_json()
        user_input = data.get("query", "").lower().strip()

        if not user_input:
            return jsonify(
                {"answer": "⚠️ Please provide a symptom or health-related question."}
            )

        greetings = {
            "hi",
            "hello",
            "hey",
            "hii",
            "good morning",
            "good afternoon",
            "good evening",
        }
        if any(greet in user_input for greet in greetings):
            return jsonify(
                {"answer": "👋 Hello! Please tell me your symptoms, and I'll try to help you."}
            )

        thanks = {"thanks", "thank you", "thx"}
        if any(thank in user_input for thank in thanks):
            return jsonify(
                {"answer": "😊 You're welcome! Let me know if you need anything else."}
            )

        bye = {"bye", "goodbye", "see you", "take care"}
        if any(farewell in user_input for farewell in bye):
            return jsonify({"answer": "👋 Take care! Wishing you good health. Bye!"})

        cleaned_input = re.sub(r"[^\w\s]", "", user_input)
        user_words = cleaned_input.split()

        all_symptoms = set()
        all_symptom_phrases = set()
        for symptom_string in df["Symptoms"]:
            all_symptom_phrases.add(symptom_string.strip().lower())
            all_symptoms.update(
                symptom.strip().lower() for symptom in symptom_string.split(",")
            )

        normalized_input_phrase = ",".join(
            sorted(
                symptom.strip().lower()
                for symptom in user_input.split(",")
                if symptom.strip()
            )
        )
        phrase_symptoms = [
            symptom.strip().lower()
            for symptom in normalized_input_phrase.split(",")
            if symptom.strip()
        ]
        extracted_symptoms = (
            phrase_symptoms
            if normalized_input_phrase in all_symptom_phrases
            else [word for word in user_words if word in all_symptoms]
        )

        if not extracted_symptoms:
            meaningful_words = [
                word for word in user_words if word not in FUZZY_MATCH_STOPWORDS
            ]
            for word in meaningful_words:
                matched_symptom = fuzzy_match_symptom(word, all_symptoms)
                if matched_symptom:
                    extracted_symptoms.append(matched_symptom)

        if not extracted_symptoms:
            return jsonify(
                {
                    "answer": "🤖 I couldn't identify any known symptoms in your message. Please try again."
                }
            )

        matched_rows = []
        for _, row in df.iterrows():
            dataset_symptoms = [
                symptom.strip().lower() for symptom in row["Symptoms"].split(",")
            ]
            if any(symptom in dataset_symptoms for symptom in extracted_symptoms):
                matched_rows.append(row)

        if matched_rows:
            matched_rows.sort(
                key=lambda row: sum(
                    symptom in row["Symptoms"].lower()
                    for symptom in extracted_symptoms
                ),
                reverse=True,
            )
            best_match = matched_rows[0]
            treatment = best_match["Treatments"]
            treatment_text = (
                MISSING_TREATMENT_MESSAGE
                if pd.isna(treatment)
                else treatment
            )
            return jsonify(
                {
                    "answer": f"🩺 Condition: {best_match['Name']}\n💊 Treatment: {treatment_text}"
                }
            )

        return jsonify(
            {"answer": "🤖 No match found for the symptoms you entered. Please try again."}
        )
    except Exception as error:
        print(f"Error: {error}")
        return jsonify({"answer": f"⚠️ An error occurred: {error}"})


@chat_bp.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Please provide a JSON request body."}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Please provide a valid JSON object."}), 400

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Please provide a non-empty query."}), 400

    try:
        return jsonify(predict_symptoms(query))
    except MLPredictionError as error:
        return jsonify({"error": str(error)}), 500
