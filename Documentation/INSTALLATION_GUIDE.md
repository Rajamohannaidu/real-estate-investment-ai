# 🚀 Installation & Setup Guide

## Complete Real Estate Investment Advisor AI

This guide will walk you through the complete installation process step-by-step.

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- **Git** installed ([Download](https://git-scm.com/downloads))
- **Groq API Key** ([Get one free](https://console.groq.com/))
- **8GB RAM** minimum (16GB recommended)
- **5GB disk space**

---

## 🛠️ Installation Methods

Choose one of these methods:

### Method 1: Automated Setup (Recommended) ⚡

```bash
# 1. Clone/download the project
cd real-estate-investment-ai

# 2. Make setup script executable
chmod +x setup.sh

# 3. Run automated setup
./setup.sh

# 4. Edit .env file with your API key
nano .env  # or use any text editor
# Change: GROQ_API_KEY=your_groq_api_key_here

# 5. Activate virtual environment
source venv/bin/activate

# 6. Run the application
streamlit run app/streamlit_app.py
```

---

### Method 2: Manual Setup 🔧

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create directories
mkdir -p data/{raw,processed}
mkdir -p models/{saved_models,explainability}
mkdir -p {logs,reports,exports}

# 6. Setup environment variables
cp .env.example .env
# Edit .env with your API key

# 7. Generate sample data
python src/data_preprocessing.py

# 8. Run application
streamlit run app/streamlit_app.py
```

---

### Method 3: Using Make 🏗️

```bash
# 1. Install everything
make install

# 2. Setup project
make setup

# 3. Edit .env with API key
nano .env

# 4. Generate data
make data

# 5. Train models (optional)
make train

# 6. Run application
make run
```

---

### Method 4: Docker 🐳

```bash
# 1. Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

# 2. Build and run
make docker-build
make docker-run

# Application will be available at http://localhost:8501

# To stop
make docker-stop
```

---

## 🔑 Getting Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to **API Keys** section
4. Click **Create API Key**
5. Copy the key
6. Paste into `.env` file:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

---

## ✅ Verify Installation

Run these commands to verify everything works:

```bash
# 1. Check Python packages
pip list | grep -E 'streamlit|langchain|shap|lime'

# 2. Test data preprocessing
python -c "from src.data_preprocessing import RealEstateDataPreprocessor; print('✓ OK')"

# 3. Test models
python -c "from src.predictive_models import RealEstatePredictiveModels; print('✓ OK')"

# 4. Test analytics
python -c "from src.investment_analytics import InvestmentAnalytics; print('✓ OK')"

# 5. Run tests
pytest tests/ -v
```

Expected output: All tests should pass ✓

---

## 🎯 First Run

1. **Activate environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Start application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Open browser:**
   - URL: `http://localhost:8501`
   - Dashboard should load automatically

4. **Try the features:**
   - Go to **Property Analysis** → Enter property details → Get predictions
   - Go to **Investment Calculator** → Calculate ROI and yields
   - Go to **AI Advisor** → Ask investment questions

---

## 📊 Generate Sample Data (Optional)

If you want to work with sample data:

```bash
# Generate 1000 sample properties
python src/data_preprocessing.py

# Or specify custom size
python -c "
from src.data_preprocessing import RealEstateDataPreprocessor
p = RealEstateDataPreprocessor()
df = p.create_sample_dataset(2000)
df.to_csv('data/sample_data.csv', index=False)
print('Generated 2000 samples')
"
```

---

## 🤖 Train Models (Optional)

To train models on your data:

```bash
# Basic training
python src/model_training.py

# With detailed report
python src/model_training.py --report

# Custom data path
python src/model_training.py --data path/to/your/data.csv
```

Training time:
- Small dataset (1K records): ~2 minutes
- Medium dataset (10K records): ~10 minutes
- Large dataset (100K+ records): 30+ minutes

---

## 🐛 Troubleshooting

### Issue: "GROQ_API_KEY not found"

**Solution:**
```bash
# Check if .env exists
ls -la .env

# If not, create it
cp .env.example .env

# Edit with your key
nano .env
```

---

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or upgrade all
pip install -r requirements.txt --upgrade
```

---

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Option 1: Kill existing process
lsof -ti:8501 | xargs kill -9

# Option 2: Use different port
streamlit run app/streamlit_app.py --server.port 8502
```

---

### Issue: TensorFlow errors on M1/M2 Mac

**Solution:**
```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal
```

---

### Issue: SHAP/LIME visualization errors

**Solution:**
```bash
# Install additional dependencies
pip install matplotlib scikit-learn --upgrade
```

---

## 🔄 Updating the Project

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations if needed
python src/model_training.py
```

---

## 🗑️ Uninstalling

```bash
# Remove virtual environment
rm -rf venv/

# Remove generated files
make clean

# Remove all data
rm -rf data/ models/ logs/ reports/ exports/
```

---

## 📦 Project Structure Reference

```
real-estate-investment-ai/
├── src/                    # Core modules
│   ├── data_preprocessing.py
│   ├── predictive_models.py
│   ├── model_training.py
│   ├── investment_analytics.py
│   ├── explainability.py
│   └── chatbot.py
│
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py
│   └── components/
│
├── tests/                  # Test suite
├── data/                   # Data files
├── models/                 # Trained models
└── requirements.txt        # Dependencies
```

---

## 📚 Additional Resources

- **Documentation**: See README.md
- **API Reference**: Check code comments
- **Examples**: View sample_data.csv
- **Tests**: Run `pytest tests/ -v`

---

## 💡 Tips

1. **Use virtual environment** - Always activate before working
2. **Keep API key secret** - Never commit .env to git
3. **Update regularly** - Run `pip install -r requirements.txt --upgrade`
4. **Monitor logs** - Check `logs/app.log` for issues
5. **Backup models** - Save trained models before retraining

---

## ✨ Next Steps

After installation:

1. ✅ Verify all features work
2. 📊 Generate or upload your data
3. 🤖 Train models on your data
4. 💬 Test the AI chatbot
5. 📈 Start analyzing properties!

---

## 🆘 Need Help?

If you encounter issues:

1. Check this guide thoroughly
2. Review error messages in terminal
3. Check `logs/app.log`
4. Run tests: `pytest tests/ -v`
5. Verify API key configuration

---

## 🎉 Success!

If you see the Streamlit dashboard, congratulations! 🎊

You're ready to start analyzing real estate investments with AI!

**Default URL**: http://localhost:8501

---

**Happy Investing! 🏘️💰**