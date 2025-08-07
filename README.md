# 💰 Smart Expense Analyser

An AI-powered financial analysis tool that automatically categorizes and analyzes bank statements from major UAE banks using advanced NLP models.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Transformers](https://img.shields.io/badge/transformers-BART--Large--MNLI-orange.svg)

---

## 🎥 Demo

**See the Smart Expense Analyser in action:**

[🎬 Watch Full Demo Video](demo/full-demo.mp4)

*Complete walkthrough: PDF Upload → Bank Detection → AI Processing → Interactive Analytics → Q&A System → PDF Report Generation*

📄 [View Sample Report](demo/sample-report.pdf)

---

## 🌟 Features

### 🏦 Multi-Bank Support
- First Abu Dhabi Bank (FAB)
- Emirates NBD
- Commercial Bank of Dubai (CBD)
- Emirates Islamic Bank (EIB)
- Mashreq Bank
- Dubai Islamic Bank (DIB)
- Abu Dhabi Commercial Bank (ADCB)
- Generic fallback parser for other banks

### 🤖 AI-Powered Categorization
- BART-Large-MNLI model for intelligent classification
- Hybrid: Rule-based + AI for high accuracy
- Real-time processing with progress updates
- Confidence scores for predictions

### 📊 Comprehensive Analytics
- Interactive dashboards with zoom and pan
- Monthly trends, balance tracking, and financial health score
- Spending breakdowns by category and timeframe

### 💬 Natural Language Q&A
Ask questions like:
- "What was my total income in August?"
- "How much did I spend on food?"
- "What's my largest expense category?"

### 📄 Professional Reporting
- Generate polished PDF reports with charts and summaries
- Monthly overviews, category details, and key metrics
- Download and share easily

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/rajrejin/Smart-Expense-Analyser.git
cd Smart-Expense-Analyser

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run expense_analyser.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Dependencies

### Core Libraries
- `streamlit>=1.28.0`
- `pandas>=1.5.0`
- `plotly>=5.15.0`
- `transformers>=4.30.0`
- `torch>=2.0.0`

### PDF Processing
- `pdfplumber>=0.9.0`
- `reportlab>=4.0.0`

### Optional (for OCR support)
- `pytesseract>=0.3.10`
- `pdf2image>=3.1.0`

---

## 💡 Usage Flow

### 1. Upload Statement
- Drag and drop your PDF
- Supports up to 100MB (configured in `.streamlit/config.toml`)
- Scanned and digital PDFs supported

### 2. Processing
- Bank detection
- Smart parsing and AI-based classification

### 3. Analytics Dashboard
- Filter, zoom, and analyze your spending
- Trends, balances, and breakdowns

### 4. Q&A
- Ask questions in natural language
- Instant financial answers powered by NLP

### 5. Generate Report
- Create a detailed PDF report
- Share with advisors or keep for records

---

## 🏗️ Architecture

### Project Structure
```text
smart-expense-analyser/
├── .streamlit/
│   └── config.toml           # Streamlit configuration (100MB upload limit)
├── demo/
│   ├── full-demo.mp4         # Complete application walkthrough
│   └── sample-report.pdf     # Example financial analysis report
├── expense_analyser.py       # Main application with all components
├── LICENSE                   # MIT License
├── README.md                 # Project documentation (you're reading this!)
├── requirements.txt          # Required Python libraries
```

### Key Modules in `expense_analyser.py`
- `ComprehensiveUAEParser`: PDF text extraction and bank parsing
- `SmartCategorizer`: Hybrid classification (rule-based + AI)
- `EnhancedQASystem`: NLP-driven financial Q&A
- `Dashboard Functions`: Plotly dashboards
- `Report Generator`: PDF summary builder

### Processing Pipeline
```
PDF → Text Extraction → Bank Detection → Transaction Parsing → 
Categorization → AI Enhancement → Dashboard & Report
```

### Supported Categories
- **Income**: Salaries, deposits
- **Expenses**: Food, transport, shopping
- **Banking**: Fees, withdrawals
- **Healthcare**: Medical, insurance
- **Business**: VAT, company charges

---

## 🛠️ Configuration

### `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 100
```

### Environment Variables (Optional)
```bash
CUDA_VISIBLE_DEVICES=0  # Use GPU if available
```

---

## 🔒 Privacy & Security

- 100% **local processing** — no data is sent to the cloud
- **Zero data storage** — nothing is saved unless you choose
- **Bank-grade privacy** — your data stays on your device

---

## 📊 Sample Outputs

### Summary
- 💰 Income: AED 25,000.00
- 💸 Expenses: AED 18,500.00
- 📈 Net Flow: AED 6,500.00
- 💳 Transactions: 156

### Category Breakdown
- Food & Dining: AED 4,200.00
- Transportation: AED 2,100.00
- Shopping: AED 3,800.00
- Utilities: AED 1,200.00

---

## 🤝 Contributing

We welcome PRs and contributions!

### How to Contribute

```bash
# Fork the repo
git clone https://github.com/yourusername/smart-expense-analyser.git
cd smart-expense-analyser

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start app
streamlit run expense_analyser.py
```

### PR Workflow
1. Fork and branch
2. Commit your changes
3. Push and open PR

---

## 🐛 Troubleshooting

**PDF parsing fails?**
```bash
pip install pytesseract pdf2image
sudo apt-get install tesseract-ocr  # (Linux)
```

**Model loading fails?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Streamlit won’t start?**
```bash
pip install --upgrade streamlit
streamlit cache clear
```

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Hugging Face Transformers
- Streamlit Framework
- Plotly for visualizations
- UAE banking community (format specifications)

---

**⭐ If you found this helpful, give the repo a star!**

Made with ❤️ for the UAE financial community.
