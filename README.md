# 💰 Smart Expense Analyser

An AI-powered financial analysis tool that automatically categorizes and analyzes bank statements from major UAE banks using advanced NLP models.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Transformers](https://img.shields.io/badge/transformers-BART--Large--MNLI-orange.svg)

## 🎥 Demo

**See the Smart Expense Analyser in action:**

[🎬 **Watch Full Demo Video**](demo/full-demo.mp4)

*Complete walkthrough: PDF Upload → Bank Detection → AI Processing → Interactive Analytics → Q&A System → PDF Report Generation*

[📄 **View Sample Report**](demo/sample-report.pdf) - See the kind of professional financial analysis this tool generates

---

## 🌟 Features

### 🏦 Multi-Bank Support
- **First Abu Dhabi Bank (FAB)**
- **Emirates NBD**
- **Commercial Bank of Dubai (CBD)**
- **Emirates Islamic Bank (EIB)**
- **Mashreq Bank**
- **Dubai Islamic Bank (DIB)**
- **Abu Dhabi Commercial Bank (ADCB)**
- Generic parser for other banks

### 🤖 AI-Powered Categorization
- **BART-Large-MNLI** model for intelligent transaction classification
- **Hybrid approach**: Rule-based + AI for maximum accuracy
- **Real-time processing** with progress tracking
- **Confidence scoring** for AI predictions

### 📊 Comprehensive Analytics
- **Interactive dashboards** with zoom/pan capabilities
- **Monthly trend analysis** and performance tracking
- **Category spending breakdowns** with visual insights
- **Balance tracking** and account growth analysis
- **Financial health scoring** and recommendations

### 💬 Natural Language Q&A
- Ask questions like: *"What was my total income in August?"*
- *"How much did I spend on food?"*
- *"What's my largest expense category?"*
- Intelligent query processing for financial insights

### 📄 Professional Reporting
- **Beautiful PDF reports** with charts and insights
- **Detailed monthly analysis** and category breakdowns
- **Key financial metrics** and recommendations
- **Exportable data** for further analysis

## 🚀 Quick Start

### Prerequisites
Python 3.8+
pip package manager

### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/smart-expense-analyser.git
cd smart-expense-analyser

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the application**
streamlit run expense_analyser.py

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📦 Dependencies

### Core Libraries
streamlit>=1.28.0          # Web interface
pandas>=1.5.0              # Data manipulation
plotly>=5.15.0             # Interactive charts
transformers>=4.30.0       # AI models
torch>=2.0.0               # Deep learning framework

### PDF Processing
pdfplumber>=0.9.0          # PDF text extraction
reportlab>=4.0.0           # PDF report generation

### Optional (Enhanced Features)
pytesseract>=0.3.10        # OCR for scanned PDFs
pdf2image>=3.1.0           # PDF to image conversion

## 💡 Usage

### 1. Upload Statement
- Drag and drop your PDF bank statement
- Supports files up to 100MB (configured in .streamlit/config.toml)
- Works with both digital and scanned PDFs

### 2. AI Processing
- Automatic bank detection
- Smart transaction parsing
- AI-powered categorization with progress tracking

### 3. Interactive Analysis
- Explore zoomable charts and trends
- Filter by categories, months, or amounts
- View detailed transaction breakdowns

### 4. Ask Questions
- Natural language queries about your finances
- Get instant answers with specific amounts and insights

### 5. Generate Reports
- Create comprehensive PDF reports
- Download professional financial analysis
- Share insights with financial advisors

## 🏗️ Architecture

### Project Structure
smart-expense-analyser/
├── .streamlit/
│   └── config.toml           # Streamlit configuration (100MB upload limit)
├── demo/
│   ├── full-demo.mp4        # Complete application walkthrough
│   └── sample-report.pdf    # Example financial analysis report
├── expense_analyser.py      # Main application with all components
├── LICENSE                  # MIT License
└── README.md               # Project documentation

### Core Components (within expense_analyser.py)
- **ComprehensiveUAEParser**: Multi-bank PDF parsing engine
- **SmartCategorizer**: Hybrid AI+Rule categorization system
- **EnhancedQASystem**: Natural language query processor
- **Dashboard Functions**: Interactive Plotly visualizations
- **Report Generator**: Professional PDF report creation

### AI Pipeline
Raw PDF → Text Extraction → Bank Detection → Transaction Parsing → 
Rule-based Categorization → AI Enhancement → Final Classification

### Supported Transaction Types
- **Income**: Salary transfers, deposits, refunds
- **Expenses**: Shopping, dining, transportation, utilities
- **Banking**: ATM withdrawals, service charges, fees
- **Healthcare**: Pharmacy, medical, insurance
- **Business**: Corporate transactions, VAT charges

### Supported Formats
- ✅ Digital PDFs with selectable text
- ✅ Scanned PDFs (with OCR enabled)
- ✅ Multi-page statements
- ✅ Multiple currencies (AED focus)

## 🛠️ Configuration

### Settings Panel
- **Max file size**: 1-100MB (default: 100MB via config.toml)
- **Max pages**: 1-30 pages (default: 20)
- **AI processing**: Enable/disable BART model

### Environment Variables
# Optional: Specify device for AI processing
CUDA_VISIBLE_DEVICES=0    # Use GPU if available

## 🔒 Privacy & Security

- **Local processing**: All data processed on your machine
- **No data storage**: Files are not saved or transmitted
- **No cloud dependency**: Works completely offline
- **Bank-grade privacy**: Your financial data stays private

## 📊 Sample Outputs

### Financial Summary
💰 Income: AED 25,000.00
💸 Expenses: AED 18,500.00
📈 Net Flow: AED 6,500.00
💳 Transactions: 156

### Category Breakdown
- **Food & Dining**: AED 4,200.00 (22.7%)
- **Transportation**: AED 2,100.00 (11.4%)
- **Shopping**: AED 3,800.00 (20.5%)
- **Utilities**: AED 1,200.00 (6.5%)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution
- **New bank parsers** for additional UAE banks
- **Enhanced AI models** for better categorization
- **Additional chart types** and visualizations
- **Multi-language support** (Arabic, Hindi, etc.)
- **Mobile optimization** for better responsive design

### Development Setup
# Fork and clone the repository
git clone https://github.com/yourusername/smart-expense-analyser.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt

# Run the application
streamlit run expense_analyser.py

### Submission Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Q: PDF parsing fails**
# Install additional dependencies
pip install pytesseract pdf2image
# For Ubuntu/Debian
sudo apt-get install tesseract-ocr

**Q: AI model loading fails**
# Check available memory and CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Reduce model complexity in settings

**Q: Streamlit won't start**
# Update Streamlit
pip install --upgrade streamlit
# Clear cache
streamlit cache clear

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** for the BART-Large-MNLI model
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **UAE Banking Community** for format specifications

**⭐ If you find this project helpful, please star the repository!**

Made with ❤️ for the UAE financial community
