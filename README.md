# 🏘️ Real Estate Investment Advisor AI

An advanced AI-powered platform for real estate investment analysis using your **Housing.csv** dataset. Combines machine learning predictions, explainable AI, and conversational intelligence to help investors make informed decisions.

## 🎯 Dataset Information

**Your Housing.csv Dataset:**
- **545 properties** with complete information
- **13 features**: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
- **Target**: Property price (in Indian Rupees)
- **Price Range**: ₹1.75M - ₹13.3M
- **Average Price**: ₹4.77M

## 🎯 Features

### 1. **Predictive Modeling**
- Multiple ML algorithms (Random Forest, XGBoost, LightGBM, Gradient Boosting)
- Deep Learning with TensorFlow/Keras
- Automated model comparison and selection
- Expected accuracy: 75-85% (R² score)
- Future price appreciation forecasting

### 2. **Investment Analytics**
- **ROI Calculator**: Calculate return on investment over custom time periods
- **Rental Yield Analysis**: Gross and net rental yield calculations
- **Cash Flow Projections**: Monthly and annual cash flow estimates
- **Cap Rate Calculation**: Property capitalization rate analysis
- **Break-even Analysis**: Investment payback period calculations
- **Comprehensive Investment Scoring**: AI-driven investment recommendations

### 3. **Explainable AI (XAI)**
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Individual prediction explanations
- Transparent decision-making process
- Feature contribution visualization

### 4. **Conversational AI Assistant**
- Powered by **LangChain** and **Groq Cloud LLM**
- Real-time investment guidance
- Natural language understanding
- Context-aware responses
- Personalized recommendations
- Property comparison assistance

### 5. **Interactive Dashboard**
- Built with **Streamlit**
- Real-time visualizations with **Plotly**
- Multi-page application architecture
- Responsive and intuitive UI

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Groq API key ([Get one here](https://console.groq.com/))

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/real-estate-investment-ai.git
cd real-estate-investment-ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_api_key_here
```

### Step 5: Generate Sample Data (Optional)
```bash
python src/data_preprocessing.py
```

### Step 6: Train Models (Optional)
```bash
python src/predictive_models.py
```

### Step 7: Run the Application
```bash
streamlit run app/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
real-estate-investment-ai/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data
│   └── sample_data.csv         # Sample dataset
│
├── models/
│   ├── saved_models/           # Trained model files
│   └── explainability/         # SHAP/LIME artifacts
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── predictive_models.py    # ML/DL model training
│   ├── investment_analytics.py # Investment calculations
│   ├── explainability.py       # SHAP and LIME implementations
│   └── chatbot.py             # LangChain chatbot
│
├── app/
│   └── streamlit_app.py       # Main Streamlit application
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🎮 Usage Guide

### 1. Property Analysis
- Navigate to **📊 Property Analysis**
- Enter property details (area, bedrooms, location, etc.)
- Click **Predict Price** to get instant predictions
- View future value projections

### 2. Investment Calculator
- Go to **💰 Investment Calculator**
- Input purchase price, rental income, and expenses
- Set your desired holding period
- Get comprehensive investment metrics:
  - ROI percentage
  - Rental yield
  - Cap rate
  - Cash flow analysis
  - Investment recommendation

### 3. Model Explainability
- Visit **🔍 Model Explainability**
- View global feature importance
- Understand individual predictions
- See SHAP value contributions

### 4. AI Investment Advisor
- Access **💬 AI Investment Advisor**
- Ask questions in natural language
- Get personalized investment advice
- Compare properties
- Receive market insights

### 5. Analytics Dashboard
- Check **📈 Dashboard**
- View portfolio analytics
- Compare multiple properties
- Analyze market trends

## 🔑 API Keys

### Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

**Supported Models:**
- `mixtral-8x7b-32768` (Default, Recommended)
- `llama2-70b-4096`
- `llama-3.1-70b-versatile`

## 📊 Sample Data Format

The system expects CSV data with the following columns:

```csv
area,bedrooms,bathrooms,year_built,location,property_type,parking_spaces,amenities_score,price
1500,3,2,2010,Urban,House,2,7,450000
2000,4,3,2015,Suburban,Villa,3,9,650000
```

## 🧪 Model Training

To train models on your own data:

```python
from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels

# Load and preprocess data
preprocessor = RealEstateDataPreprocessor()
df = preprocessor.load_data('data/your_data.csv')
df = preprocessor.clean_data(df)
df = preprocessor.feature_engineering(df)

# Encode categorical variables
categorical_cols = ['location', 'property_type']
df_encoded = preprocessor.encode_categorical(df, categorical_cols)

# Prepare features
X_train, X_test, y_train, y_test = preprocessor.prepare_features(df_encoded)

# Train models
models = RealEstatePredictiveModels()
results = models.train_all_models(X_train, y_train, X_test, y_test)

# Save models
models.save_models()
```

## 🤖 Chatbot Customization

Customize the AI assistant by modifying the system prompt in `src/chatbot.py`:

```python
def _create_system_prompt(self):
    return """Your custom investment advisor prompt here..."""
```

## 📈 Investment Metrics Explained

### ROI (Return on Investment)
```
ROI = (Net Profit / Total Investment) × 100
```

### Rental Yield
```
Gross Yield = (Annual Rental Income / Property Price) × 100
Net Yield = (Net Annual Income / Property Price) × 100
```

### Cap Rate (Capitalization Rate)
```
Cap Rate = (Net Operating Income / Property Price) × 100
```

### Cash Flow
```
Cash Flow = Rental Income - All Expenses
```

## 🛠️ Troubleshooting

### Common Issues

**1. Groq API Error**
- Verify your API key is correct in `.env`
- Check your internet connection
- Ensure you have API credits

**2. Model Loading Issues**
- Run `python src/predictive_models.py` to train models
- Check `models/saved_models/` directory exists

**3. Data Loading Errors**
- Ensure CSV format matches expected columns
- Check for missing values in critical columns

**4. Installation Problems**
```bash
# Upgrade pip
pip install --upgrade pip

# Install with specific versions
pip install -r requirements.txt --no-cache-dir
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the conversational AI framework
- **Groq**: For fast LLM inference
- **SHAP & LIME**: For model explainability
- **Streamlit**: For the interactive dashboard
- **Scikit-learn & TensorFlow**: For ML/DL capabilities

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🔮 Future Enhancements

- [ ] Integration with real estate APIs (Zillow, Redfin)
- [ ] Mobile application
- [ ] Multi-user authentication
- [ ] Portfolio management features
- [ ] Advanced market trend analysis
- [ ] Integration with mortgage calculators
- [ ] Property comparison tool enhancements
- [ ] PDF report generation
- [ ] Email notifications for investment opportunities

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Built with ❤️ for Real Estate Investors**