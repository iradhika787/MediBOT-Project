# MediBOT

**AI-Powered Healthcare Assistant with Symptom-Based Machine Learning Predictions**

## Overview

MediBOT is a web-based healthcare assistance project that combines a symptom-matching chatbot with a machine learning-based disease prediction system.

### Features

- Symptom-based chatbot providing condition and treatment information.
- Machine learning model generating the top three disease predictions.
- Voice input and speech output.
- Doctor appointment scheduling interface.
- Medication reminder interface.
- Responsive web interface.

**Disclaimer:** MediBOT is an educational project. Its predictions are experimental and must not be considered a medical diagnosis or a substitute for professional medical advice.

## Technology Stack

- **Backend:** Python, Flask, Flask-CORS
- **Frontend:** HTML, CSS, JavaScript
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, SciPy, Joblib
- **Chatbot Matching:** FuzzyWuzzy
- **Voice Features:** Browser Speech Recognition and Text-to-Speech APIs

## Project Structure

```text
MediBot/
â”œâ”€â”€ app.py
â”œâ”€â”€ index.html
â”œâ”€â”€ features.html
â”œâ”€â”€ chatbot.html
â”œâ”€â”€ APP.HTML
â”œâ”€â”€ med.html
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ data/
â”‚   â””â”€â”€ medical_ds.csv
â”œâ”€â”€ medibot/
â”‚   â”œâ”€â”€ routes/
â”‚   â”œâ”€â”€ services/
â”‚   â””â”€â”€ utils/
â”œâ”€â”€ ml/
â”‚   â””â”€â”€ train_models.py
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ best_model.joblib
â”‚   â””â”€â”€ feature_metadata.json
â””â”€â”€ reports/
    â”œâ”€â”€ model_comparison.csv
    â”œâ”€â”€ final_test_metrics.json
    â”œâ”€â”€ final_classification_report.csv
    â”œâ”€â”€ final_confusion_matrix.csv
    â””â”€â”€ final_confusion_matrix.png
```

## API Endpoints

### 1. Symptom Chatbot

`POST /chat`

Accepts a symptom-related query and returns a matching condition and available treatment information.

Example request:

```json
{
  "query": "dizziness"
}
```

### 2. Machine Learning Prediction

`POST /predict`

Accepts symptoms and returns up to three disease predictions based on the trained machine learning model.

Example request:

```json
{
  "query": "dizziness and shortness of breath"
}
```

The response includes recognized symptoms and model scores. These scores are not calibrated probabilities.

## Machine Learning

The prediction system uses a cleaned and processed augmented symptom-disease dataset.

### Model Evaluation

Three classification algorithms were compared:

- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

Logistic Regression was selected based on validation macro F1-score.

### Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 95.08% |
| Macro Precision | 94.48% |
| Macro Recall | 96.42% |
| Macro F1-score | 95.27% |
| Weighted F1-score | 95.14% |
| Balanced Accuracy | 96.42% |

Evaluation was performed on a held-out test set of 22,903 records. These results measure performance on the prepared dataset and do not establish real-world clinical accuracy.

## Installation and Setup

### Prerequisites

- Python 3.11
- pip

### Steps

1. Clone the repository:

```bash
git clone https://github.com/iradhika787/MediBOT-Project.git
```

2. Navigate to the project directory:

```bash
cd MediBOT-Project
```

3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Start the Flask application:

```bash
python app.py
```

5. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Disclaimer

MediBOT is developed for educational and demonstration purposes only. Machine learning predictions are experimental and should not be used for diagnosis, treatment decisions, or emergencies. Always consult qualified healthcare professionals.
