# ✅ Setup Checklist for Housing.csv Project

Use this checklist to ensure everything is properly configured.

---

## 📋 Pre-Installation Checklist

- [ ] Python 3.8+ installed (`python --version`)
- [ ] pip installed and updated (`pip --version`)
- [ ] Housing.csv file available
- [ ] Git installed (optional, for version control)
- [ ] 8GB+ RAM available
- [ ] 2GB+ disk space available

---

## 🔧 Installation Checklist

### Step 1: Project Setup
- [ ] Created project directory
- [ ] Placed Housing.csv in project root
- [ ] Created virtual environment (`python -m venv venv`)
- [ ] Activated virtual environment
- [ ] Verified activation (prompt shows `(venv)`)

### Step 2: Dependencies
- [ ] requirements.txt file present
- [ ] Ran `pip install -r requirements.txt`
- [ ] No installation errors
- [ ] Key packages installed:
  - [ ] streamlit
  - [ ] pandas
  - [ ] scikit-learn
  - [ ] tensorflow
  - [ ] xgboost
  - [ ] lightgbm
  - [ ] shap
  - [ ] lime
  - [ ] langchain
  - [ ] plotly

### Step 3: Directory Structure
- [ ] `src/` directory exists
- [ ] `app/` directory exists
- [ ] `models/` directory exists
- [ ] `data/` directory exists
- [ ] `tests/` directory exists
- [ ] All __init__.py files present

### Step 4: Configuration Files
- [ ] `.env` file created
- [ ] GROQ_API_KEY added to .env (if using chatbot)
- [ ] config.py present
- [ ] utils.py present

---

## 🤖 Model Training Checklist

### Before Training
- [ ] Housing.csv in correct location
- [ ] File has 545 records
- [ ] All 13 columns present
- [ ] No missing values in file
- [ ] train_housing_models.py script present

### Training Process
- [ ] Ran `python train_housing_models.py`
- [ ] Data loaded successfully (545 records)
- [ ] Binary conversion completed
- [ ] Feature engineering completed
- [ ] All 7 models trained
- [ ] No errors during training
- [ ] Best model identified
- [ ] Models saved to `models/saved_models/`

### Training Verification
- [ ] R² Score > 0.70 (preferably > 0.75)
- [ ] RMSE < ₹800,000 (preferably < ₹600,000)
- [ ] training_results.json created
- [ ] At least 5 model files (.pkl) saved
- [ ] scaler.pkl saved
- [ ] feature_names.pkl saved

---

## 🚀 Application Launch Checklist

### Pre-Launch
- [ ] Models trained and saved
- [ ] streamlit installed
- [ ] app/streamlit_app.py exists
- [ ] Port 8501 available

### Launch
- [ ] Ran `streamlit run app/streamlit_app.py`
- [ ] No errors in terminal
- [ ] Browser opened automatically
- [ ] App loads at localhost:8501

### UI Verification
- [ ] Home page displays
- [ ] Navigation sidebar visible
- [ ] 6 pages accessible:
  - [ ] 🏠 Home
  - [ ] 📊 Property Analysis
  - [ ] 💰 Investment Calculator
  - [ ] 🔍 Model Explainability
  - [ ] 💬 AI Investment Advisor
  - [ ] 📈 Dashboard

---

## 🧪 Feature Testing Checklist

### Property Analysis Page
- [ ] Can enter area (1650-16200)
- [ ] Can select bedrooms (1-6)
- [ ] Can select bathrooms (1-4)
- [ ] Can select stories (1-4)
- [ ] All yes/no options work
- [ ] Furnishing dropdown works
- [ ] "Predict Price" button works
- [ ] Prediction displays
- [ ] Price is in reasonable range (₹1.5M - ₹15M)
- [ ] Future projections chart displays
- [ ] No errors in console

### Investment Calculator Page
- [ ] Can enter purchase price
- [ ] Can enter rental income
- [ ] Can enter expenses
- [ ] Calculate button works
- [ ] ROI displays correctly
- [ ] Rental yield shows
- [ ] Cap rate shows
- [ ] Cash flow shows
- [ ] Recommendation appears
- [ ] Charts render correctly

### Model Explainability Page
- [ ] Feature importance chart displays
- [ ] SHAP explanation section visible
- [ ] LIME explanation section visible
- [ ] Charts are interactive
- [ ] No errors

### AI Advisor Page (Optional)
- [ ] Chat interface visible
- [ ] Can type messages
- [ ] Send button works
- [ ] AI responses appear (if API key configured)
- [ ] Quick questions work
- [ ] Clear chat button works
- [ ] OR shows API key error message (if not configured)

### Dashboard Page
- [ ] Portfolio metrics display
- [ ] Charts render
- [ ] Data table shows
- [ ] All visualizations interactive
- [ ] No errors

---

## 🔑 API Configuration Checklist (Optional)

For AI Chatbot feature:

- [ ] Visited https://console.groq.com/
- [ ] Created account
- [ ] Generated API key
- [ ] Copied API key
- [ ] Created .env file
- [ ] Added: `GROQ_API_KEY=your_key_here`
- [ ] Restarted Streamlit app
- [ ] Chatbot now works
- [ ] Can ask questions
- [ ] Receives responses

---

## 🐛 Troubleshooting Checklist

If something doesn't work:

### Data Issues
- [ ] Housing.csv in project root (not in subdirectory)
- [ ] File name is exact: Housing.csv (case-sensitive)
- [ ] File has 545 rows + 1 header row
- [ ] File is CSV format (not Excel)
- [ ] No special characters in path

### Installation Issues
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] Ran `pip install --upgrade pip`
- [ ] Tried `pip install -r requirements.txt --upgrade`
- [ ] No network/firewall issues
- [ ] Python version correct (3.8+)

### Training Issues
- [ ] In project root directory
- [ ] Housing.csv present
- [ ] Enough disk space (2GB+)
- [ ] Enough RAM (8GB+)
- [ ] No other Python processes using models
- [ ] Try deleting models/ folder and retrain

### App Issues
- [ ] Port 8501 not used by other app
- [ ] Models trained first
- [ ] No errors in terminal
- [ ] Browser not blocking localhost
- [ ] Try different browser
- [ ] Try `streamlit run app/streamlit_app.py --server.port 8502`

### Performance Issues
- [ ] Dataset size appropriate (545 is good)
- [ ] Models trained successfully
- [ ] R² score acceptable (>0.70)
- [ ] Predictions reasonable
- [ ] Charts render within 5 seconds

---

## ✨ Success Indicators

You know everything is working when:

### Training Success
✅ Training completes in 2-5 minutes  
✅ R² Score between 0.75-0.85  
✅ RMSE between ₹400k-₹600k  
✅ All 7 models saved  
✅ No error messages  

### App Success
✅ App opens in browser  
✅ All pages accessible  
✅ Predictions work  
✅ Charts display  
✅ No console errors  

### Prediction Success
✅ Prices in ₹2M-₹12M range  
✅ Similar properties get similar prices  
✅ Larger area = higher price  
✅ More bedrooms = higher price  
✅ Predictions within ±20% of expected  

### Investment Analysis Success
✅ ROI calculations make sense  
✅ Rental yield is 3-8%  
✅ Cash flow can be positive/negative  
✅ Recommendations are logical  

---

## 📊 Expected Results

### Model Performance
```
Best Model: Random Forest or XGBoost
R² Score: 0.78-0.82
RMSE: ₹450,000-₹550,000
MAE: ₹350,000-₹450,000
Training Time: 2-4 minutes
```

### Prediction Accuracy
```
Within ₹300k: ~60% of predictions
Within ₹500k: ~75% of predictions
Within ₹800k: ~90% of predictions
```

### App Performance
```
Load Time: 2-3 seconds
Prediction Time: <1 second
Chart Render: 1-2 seconds
Memory Usage: 500MB-1GB
```

---

## 🎯 Final Verification

Run these commands to verify everything:

```bash
# Check Python packages
pip list | grep -E 'streamlit|scikit|tensorflow|xgboost'

# Verify Housing.csv
wc -l Housing.csv  # Should show 546 (545 + header)

# Check models
ls models/saved_models/  # Should show .pkl files

# Test training
python train_housing_models.py

# Launch app
streamlit run app/streamlit_app.py
```

---

## 📝 Notes

- ⚠️ Training time depends on your CPU (2-10 minutes normal)
- ⚠️ First Streamlit run may be slower (caching)
- ⚠️ Chatbot requires API key (optional feature)
- ⚠️ Models need retraining if data changes
- ℹ️ R² Score of 0.75+ is excellent for real estate
- ℹ️ Predictions ±₹500k are very good
- ℹ️ App works offline (except chatbot)

---

## ✅ You're Done When...

- [x] All checklist items completed
- [x] No error messages
- [x] App runs smoothly
- [x] Predictions are reasonable
- [x] All features accessible
- [x] Ready to analyze properties!

---

**🎉 Congratulations! Your Real Estate Investment Advisor AI is ready!**

Next: Start analyzing properties and making investment decisions!