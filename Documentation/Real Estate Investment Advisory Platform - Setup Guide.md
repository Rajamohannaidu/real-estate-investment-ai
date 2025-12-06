# Real Estate Investment Advisory Platform - Setup Guide

## Project Structure
```
real-estate-investment-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data.csv
├── models/
│   ├── saved_models/
│   └── explainability/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── predictive_models.py
│   ├── investment_analytics.py
│   ├── explainability.py
│   └── chatbot.py
├── app/
│   ├── streamlit_app.py
│   └── components/
│       ├── prediction_view.py
│       ├── analytics_view.py
│       ├── explainability_view.py
│       └── chatbot_view.py
├── requirements.txt
├── .env.example
└── README.md
```

## Installation Steps

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run app/streamlit_app.py
```

## Key Features Implemented

1. **Predictive Modeling**: ML/DL models for price forecasting
2. **Investment Analytics**: ROI, rental yield, appreciation calculators
3. **Explainable AI**: SHAP & LIME integration
4. **Conversational AI**: LangChain + Groq LLM chatbot
5. **Interactive Dashboard**: Streamlit-based visualization

## Next Steps

1. Obtain Groq API key from: https://console.groq.com/
2. Prepare your real estate dataset
3. Train models using the provided scripts
4. Launch the Streamlit dashboard