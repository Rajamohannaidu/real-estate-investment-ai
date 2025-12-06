#!/bin/bash

# Real Estate Investment Advisor - Setup Script
# This script automates the complete project setup

set -e  # Exit on error

echo "=================================================="
echo "Real Estate Investment Advisor - Setup Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Python is installed
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 8 ]; then
    print_error "Python 3.8 or higher is required. Found Python $PYTHON_MAJOR.$PYTHON_MINOR"
    exit 1
fi

print_success "Python version is compatible"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "All dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/saved_models
mkdir -p models/explainability
mkdir -p app/components
mkdir -p logs
mkdir -p reports
mkdir -p exports
mkdir -p tests

print_success "Directory structure created"
echo ""

# Create .env file
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_success ".env file created from template"
        print_info "Please edit .env and add your GROQ_API_KEY"
    else
        cat > .env << EOF
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Model Configuration
GROQ_MODEL=mixtral-8x7b-32768

# Application Settings
DEBUG_MODE=False
DATA_PATH=data/
MODEL_PATH=models/saved_models/
EOF
        print_success ".env file created"
        print_info "Please edit .env and add your GROQ_API_KEY"
    fi
else
    print_info ".env file already exists"
fi
echo ""

# Create __init__.py files
echo "Creating Python package files..."
touch src/__init__.py
touch app/__init__.py
touch app/components/__init__.py
touch tests/__init__.py
print_success "Package files created"
echo ""

# Generate sample data
echo "Generating sample dataset..."
if [ -f "src/data_preprocessing.py" ]; then
    python src/data_preprocessing.py
    print_success "Sample dataset generated"
else
    print_info "data_preprocessing.py not found, skipping sample data generation"
fi
echo ""

# Check for GROQ API key
echo "Checking API configuration..."
if grep -q "your_groq_api_key_here" .env; then
    print_error "GROQ_API_KEY not configured in .env file"
    echo ""
    echo "To complete setup:"
    echo "1. Visit https://console.groq.com/"
    echo "2. Create an account and generate an API key"
    echo "3. Edit .env file and replace 'your_groq_api_key_here' with your actual API key"
    echo ""
else
    print_success "API key appears to be configured"
fi
echo ""

# Display final instructions
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure your Groq API key in .env file"
echo ""
echo "3. Generate sample data (if not done):"
echo "   python src/data_preprocessing.py"
echo ""
echo "4. Train models (optional):"
echo "   python src/model_training.py --report"
echo ""
echo "5. Run the application:"
echo "   streamlit run app/streamlit_app.py"
echo ""
echo "6. Or use make commands:"
echo "   make data    # Generate sample data"
echo "   make train   # Train models"
echo "   make run     # Run application"
echo ""
echo "For Docker deployment:"
echo "   make docker-build"
echo "   make docker-run"
echo ""
echo "=================================================="
echo ""

print_success "All done! Happy investing! 🏘️"