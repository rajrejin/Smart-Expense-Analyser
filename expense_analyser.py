# ================= IMPORTS & SETUP ===================
import streamlit as st, pandas as pd, re, io, torch
import plotly.graph_objects as go, plotly.express as px
from datetime import datetime
from transformers import pipeline
import pdfplumber
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

# Optional OCR support for scanned PDFs
try:
    import pytesseract, pdf2image
    ENHANCED_FEATURES = True
except ImportError:
    ENHANCED_FEATURES = False

# ================= AI MODEL LOADING ===================
@st.cache_resource
def load_bart_classifier():
    """Load BART-Large-MNLI for advanced transaction categorization"""
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", 
                       device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.warning(f"BART model loading failed, using rule-based system: {e}")
        return None

# ================= PDF PARSER CLASS ===================
class ComprehensiveUAEParser:
    """Main parser class that handles all UAE banks with text processing support"""
    
    def __init__(self, file_content, max_pages=20, max_file_size=50):
        self.file_content, self.max_pages, self.max_file_size, self.detected_bank = file_content, max_pages, max_file_size, None
        
        # Regex patterns for extracting amounts in different formats
        self.amount_patterns = {
            'standard': r'\d{1,3}(?:,\d{3})*\.\d{2}', 'aed_format': r'AED\s*[+-]?\d{1,3}(?:,\d{3})*\.\d{2}',
            'large_amounts': r'\d{1,3}(?:,\d{3})*\.\d{2}', 'simple': r'\d+\.\d{2}', 'negative_format': r'-\d{1,3}(?:,\d{3})*\.\d{2}'
        }        
        # Date patterns specific to each bank's format
        self.date_patterns = {
            'fab': [r'\d{2}-[A-Z][a-z]{2}-\d{4}', r'\d{1,2}-\w{3}-\d{4}'],
            'enbd': [r'\d{2}/\d{2}/\d{4}', r'\d{1,2}/\d{1,2}/\d{4}'],
            'cbd': [r'\d{2}/\d{2}/\d{4}', r'\d{1,2}/\d{1,2}/\d{4}'],
            'eib': [r'\d{2}-\d{2}-\d{4}', r'\d{1,2}-\d{1,2}-\d{4}'],
            'mashreq': [r'\d{2}/\d{2}/\d{4}', r'\d{1,2}/\d{1,2}/\d{4}'],
            'dib': [r'\d{2}/\d{2}/\d{4}', r'\d{1,2}/\d{1,2}/\d{4}'],
            'generic': [r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', r'\d{1,2}[-/]\w{3}[-/]\d{2,4}']
        }        
        # Bank identification keywords
        self.banks = {
            'fab': {'name': 'First Abu Dhabi Bank', 'ids': ['first abu dhabi', 'fab', 'bankfab.com']},
            'enbd': {'name': 'Emirates NBD', 'ids': ['emirates nbd', 'enbd', 'emiratesnbd']},
            'cbd': {'name': 'Commercial Bank of Dubai', 'ids': ['commercial bank of dubai', 'cbd psc', 'cbd.ae']},
            'eib': {'name': 'Emirates Islamic Bank', 'ids': ['emirates islamic', 'emiratesislamic', 'account statement']},
            'mashreq': {'name': 'Mashreq Bank', 'ids': ['mashreq', 'mashreq psc']},
            'dib': {'name': 'Dubai Islamic Bank', 'ids': ['dubai islamic', 'dib', 'dibindia']},
            'adcb': {'name': 'Abu Dhabi Commercial Bank', 'ids': ['adcb']}
        }

    def _detect_bank(self, content):
        """Auto-detect bank from PDF content using keywords"""
        content = content.lower()
        for bank_key, config in self.banks.items():
            if any(id_str in content for id_str in config['ids']):
                self.detected_bank = bank_key
                st.info(f"🏦 {config['name']} detected")
                return bank_key
        st.info("🏦 Using smart parsing for your bank")
        return 'generic'

    def _extract_amounts_by_bank(self, text, bank_type='generic'):
        """Extract monetary amounts using bank-specific patterns"""
        amounts = []
        pattern = self.amount_patterns['aed_format'] if bank_type == 'enbd' else self.amount_patterns['large_amounts'] if bank_type == 'cbd' else self.amount_patterns['standard']
        
        for match in re.finditer(pattern, text):
            try:
                amount_str = match.group()
                is_negative = '-' in amount_str or amount_str.startswith('AED -')
                clean = re.sub(r'[^\d.,-]', '', amount_str).replace(',', '')
                amount = float(clean)
                amounts.append(-abs(amount) if is_negative else amount)
            except: continue
        return amounts

    # Bank-specific line parsers with specialized logic
    def _parse_fab_line(self, line):
        """Parse First Abu Dhabi Bank transaction lines"""
        line = line.strip()
        if len(line) < 20: return None        
        date_match = next((re.search(pattern, line).group() for pattern in self.date_patterns['fab'] if re.search(pattern, line)), None)
        if not date_match: return None        
        amounts = self._extract_amounts_by_bank(line, 'fab')
        if len(amounts) < 2: return None        
        desc = line
        for pattern in self.date_patterns['fab']: desc = re.sub(pattern, '', desc)
        desc = ' '.join(re.sub(r'\d{1,3}(?:,\d{3})*\.\d{2}', '', desc).split()).strip()
        if not desc or len(desc) < 5: desc = 'Transaction'        
        is_salary = 'salary transfer' in desc.lower() and 'keystone' in desc.lower()
        is_debit_card = 'debit card transaction' in desc.lower()
        is_atm = 'atm cash withdrawal' in desc.lower()
        balance = amounts[-1]
        
        if is_salary:
            credit_amount = next((amt for amt in amounts if amt == 12200.0), amounts[0])
            return {'Date': date_match, 'Transaction': desc, 'Debit': 0, 'Credit': credit_amount, 'Balance': balance}
        elif is_debit_card or is_atm:
            debit_amount = min([amt for amt in amounts[:-1] if amt > 0]) if len(amounts) > 1 else amounts[0]
            return {'Date': date_match, 'Transaction': desc, 'Debit': debit_amount, 'Credit': 0, 'Balance': balance}
        else:
            transaction_amount = amounts[0] if amounts[0] != balance else (amounts[1] if len(amounts) > 1 else amounts[0])
            return {'Date': date_match, 'Transaction': desc, 'Debit': transaction_amount if transaction_amount > 0 else 0, 'Credit': 0 if transaction_amount > 0 else abs(transaction_amount), 'Balance': balance}

    def _parse_enbd_line(self, line):
        """Parse Emirates NBD transaction lines with AED prefix handling"""
        line = line.strip()
        if len(line) < 15: return None        
        date_match = next((re.search(pattern, line).group() for pattern in self.date_patterns['enbd'] if re.search(pattern, line)), None)
        if not date_match: return None        
        amounts, matches = [], re.findall(r'AED\s*([+-]?)(\d{1,3}(?:,\d{3})*\.\d{2})', line)        
        if not matches: return None        
        for sign, amount_str in matches:
            try:
                amount = float(amount_str.replace(',', ''))
                amounts.append(-amount if sign == '-' or 'AED -' in line else amount)
            except: continue        
        if len(amounts) < 2: return None        
        desc = line
        for pattern in self.date_patterns['enbd']: desc = re.sub(pattern, '', desc)
        desc = ' '.join(re.sub(r'AED\s*[+-]?\d{1,3}(?:,\d{3})*\.\d{2}', '', desc).split()).strip()
        if not desc: desc = 'Transaction'        
        transaction_amount, balance = amounts[0], amounts[1]        
        return {'Date': date_match, 'Transaction': desc, 'Debit': abs(transaction_amount) if transaction_amount < 0 else 0, 'Credit': transaction_amount if transaction_amount > 0 else 0, 'Balance': balance}

    def _parse_eib_line(self, line):
        """Parse Emirates Islamic Bank transaction lines with reference masking"""
        line = line.strip()
        if len(line) < 20: return None        
        date_match = next((re.search(pattern, line).group() for pattern in self.date_patterns['eib'] if re.search(pattern, line)), None)
        if not date_match: return None        
        amounts = self._extract_amounts_by_bank(line, 'eib')
        if len(amounts) < 2: return None        
        desc = line
        for pattern in self.date_patterns['eib']: desc = re.sub(pattern, '', desc)
        desc = re.sub(r'480666\w+', 'XXXX', re.sub(r'\d{10,}', '[REF]', re.sub(r'\d{1,3}(?:,\d{3})*\.\d{2}', '', desc)))
        desc = ' '.join(desc.split()).strip()
        if not desc: desc = 'Transaction'        
        balance = amounts[-1]        
        if len(amounts) >= 3:
            credit_amount, debit_amount = amounts[0], amounts[1]
            return {'Date': date_match, 'Transaction': desc, 'Debit': debit_amount if debit_amount > 0 else 0, 'Credit': credit_amount if credit_amount > 0 else 0, 'Balance': balance}
        else:
            is_salary = 'transfer salary' in desc.lower()
            is_debit = any(word in desc.lower() for word in ['pos', 'withdrawal', 'atm', 'service charges'])            
            transaction_amount = amounts[0] if amounts[0] != balance else amounts[1] if len(amounts) > 1 else amounts[0]            
            if is_salary:
                return {'Date': date_match, 'Transaction': desc, 'Debit': 0, 'Credit': transaction_amount, 'Balance': balance}
            elif is_debit:
                return {'Date': date_match, 'Transaction': desc, 'Debit': transaction_amount, 'Credit': 0, 'Balance': balance}
            else:
                return {'Date': date_match, 'Transaction': desc, 'Debit': 0, 'Credit': transaction_amount, 'Balance': balance}

    def _parse_generic_line(self, line):
        """Generic parser for unrecognized banks"""
        line = line.strip()
        if len(line) < 10: return None        
        date_match = next((re.search(pattern, line).group() for pattern in self.date_patterns['generic'] if re.search(pattern, line)), None)
        if not date_match: return None        
        amounts = self._extract_amounts_by_bank(line, 'generic')
        if not amounts: return None        
        desc = line
        for pattern in self.date_patterns['generic']: desc = re.sub(pattern, '', desc)
        desc = ' '.join(re.sub(r'\d{1,3}(?:,\d{3})*\.\d{2}', '', desc).split()).strip()
        if not desc: desc = 'Transaction'        
        is_salary = any(kw in desc.lower() for kw in ['salary', 'transfer salary'])        
        if len(amounts) >= 2:
            balance, transaction_amount = amounts[-1], amounts[0]
            return {'Date': date_match, 'Transaction': desc, 'Debit': 0 if is_salary else transaction_amount, 'Credit': transaction_amount if is_salary else 0, 'Balance': balance}
        else:
            return {'Date': date_match, 'Transaction': desc, 'Debit': 0 if is_salary else amounts[0], 'Credit': amounts[0] if is_salary else 0, 'Balance': None}

    def parse(self):
        """Main parsing function - extracts and processes all transactions"""
        try:
            # Check file size limit
            if len(self.file_content) / (1024 * 1024) > self.max_file_size:
                st.error(f"File exceeds {self.max_file_size} MB limit")
                return pd.DataFrame()            
            with pdfplumber.open(io.BytesIO(self.file_content)) as pdf:
                # Extract text for bank detection (first 3 pages)
                detection_text = "".join([page.extract_text() + "\n" for page in pdf.pages[:3] if page.extract_text()])
                self._detect_bank(detection_text)
                # Extract full text for transaction parsing
                full_text = "".join([page.extract_text() + "\n" for page in pdf.pages[:self.max_pages] if page.extract_text()])
                transactions = []
                # Parse each line using appropriate bank parser
                for line in full_text.split('\n'):
                    if self.detected_bank == 'fab': parsed = self._parse_fab_line(line)
                    elif self.detected_bank == 'enbd': parsed = self._parse_enbd_line(line)
                    elif self.detected_bank == 'eib': parsed = self._parse_eib_line(line)
                    else: parsed = self._parse_generic_line(line)
                    if parsed: transactions.append(parsed)
                df = pd.DataFrame(transactions)            
            if df.empty:
                st.warning("No transactions found. Please check if the PDF has selectable text.")
                return df            
            # Data cleaning and type conversion
            for col in ['Debit', 'Credit']: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
            # Calculate derived fields
            df['Net_Amount'] = df['Credit'] - df['Debit']
            df['Bank'] = self.banks.get(self.detected_bank, {'name': 'Unknown Bank'})['name']            
            total_income, total_expenses = df['Credit'].sum(), df['Debit'].sum()            
            st.success(f"✅ Successfully parsed {len(df)} transactions")
            st.info(f"💰 Income: AED {total_income:,.2f} | Expenses: AED {total_expenses:,.2f} | Net: AED {df['Net_Amount'].sum():,.2f}")
            return df            
        except Exception as e:
            st.error(f"Parsing failed: {e}")
            return pd.DataFrame()

# ================= SMART CATEGORIZER ===================
class SmartCategorizer:
    """Hybrid categorization using rules + AI for better accuracy"""
    
    def __init__(self, classifier=None):
        self.classifier = classifier
        # Category keywords for rule-based classification - ORIGINAL
        self.categories = {
            'Food & Dining': ['carrefour', 'lulu', 'spinneys', 'waitrose', 'metro manila', 'al maya', 'union coop', 'shams al madina', 'hippo box', 'al bakr fresh', 'daylife', 'laloco', 'tresind', 'babel', 'noon restaurant', 'claw bbq', 'bb social', 'buffalo wings', 'czn burak', 'amazonico', 'ave dubai', 'boca', 'kulture house', 'restaurant', 'cafe', 'food'],
            'Transportation': ['cars taxi', 'enoc', 'fuel', 'petrol', 'taxi', 'uber', 'careem', 'salik', 'adnoc', 'eppco'],
            'Shopping': ['zara', 'grand shopping', 'shopping mall', 'mall', 'store', 'h&m', 'ikea', 'noon'],
            'ATM & Cash': ['atm cash withdrawal', 'atm', 'cash', 'atm/ccdm cash deposit', 'ccdm', 'withdrawal'],
            'Bank Services': ['ipp charges', 'charges', 'fee', 'service', 'guarantee issue', 'vat charges', 'cheque return'],
            'Utilities': ['du apple pay', 'du payment', 'etisalat', 'dewa', 'electricity', 'water', 'internet'],
            'Healthcare': ['grand united pharmacy', 'life pharmacy', 'four seasons pharmacy', 'aster pharmacy', '800 pharmacy', 'asia pharmacy', 'pharmacy', 'hospital', 'medical', 'clinic'],
            'Income': ['salary transfer', 'keystone group', 'transfer salary', 'salary credit', 'salary dibfts', 'cheque deposit', 'deposit'],
            'Business': ['guarantee issue commission', 'vat', 'business', 'corporate'],
            'Other': []
        }

    def categorize(self, df, progress_callback=None):  # ADDED: progress_callback parameter
        """Apply rule-based then AI categorization"""
        if df.empty: return df        
        def get_category(transaction):  # ORIGINAL
            trans_lower = str(transaction).lower()
            for category, keywords in self.categories.items():
                if any(keyword in trans_lower for keyword in keywords): return category
            return 'Other'        
        # Rule-based categorization (fast and accurate for known patterns) - ORIGINAL
        df['Category'] = df['Transaction'].apply(get_category)        
        
        # ADDED: AI processing detailed tracking
        ai_processing_details = []
        ai_stats = {
            'total_processed': 0,
            'successfully_categorized': 0,
            'remained_other': 0
        }
        
        # AI-based categorization for uncategorized transactions - ORIGINAL LOGIC
        if self.classifier:
            other_mask = df['Category'] == 'Other'
            if other_mask.sum() > 0:
                category_labels = [cat for cat in self.categories.keys() if cat != 'Other']                
                
                # ADDED: Statistics setup
                ai_indices = df[other_mask].index.tolist()
                total_ai = len(ai_indices)
                ai_stats['total_processed'] = total_ai
                
                for current, idx in enumerate(ai_indices, 1):  # MODIFIED: added enumeration for progress
                    try:
                        transaction = df.loc[idx, 'Transaction']  # ORIGINAL
                        result = self.classifier(transaction, category_labels)  # ORIGINAL
                        confidence_score = result['scores'][0]
                        predicted_category = result['labels'][0]
                        
                        if confidence_score > 0.6:  # ORIGINAL confidence threshold
                            df.loc[idx, 'Category'] = predicted_category
                            ai_stats['successfully_categorized'] += 1
                            status = "✅ Success"
                            final_category = predicted_category
                        else:
                            ai_stats['remained_other'] += 1
                            status = "❌ Failed"
                            final_category = "Other"
                        
                        # ADDED: Store detailed transaction info
                        ai_processing_details.append({
                            'Transaction': transaction[:60] + "..." if len(transaction) > 60 else transaction,
                            'AI Prediction': predicted_category,
                            'Confidence': f"{confidence_score:.2%}",
                            'Status': status,
                            'Final Category': final_category,
                            'Date': df.loc[idx, 'Date'].strftime('%Y-%m-%d') if 'Date' in df.columns else 'N/A',
                            'Amount': f"AED {df.loc[idx, 'Debit']:.2f}" if df.loc[idx, 'Debit'] > 0 else f"AED {df.loc[idx, 'Credit']:.2f}"
                        })
                        
                        # ADDED: Progress callback
                        if progress_callback:
                            progress_callback(current, total_ai)
                            
                    except Exception as e:  # ORIGINAL
                        ai_stats['remained_other'] += 1  # ADDED: Track exceptions
                        
                        # ADDED: Store failed transaction info
                        ai_processing_details.append({
                            'Transaction': transaction[:60] + "..." if len(transaction) > 60 else transaction,
                            'AI Prediction': 'Error',
                            'Confidence': '0.00%',
                            'Status': '⚠️ Error',
                            'Final Category': 'Other',
                            'Date': df.loc[idx, 'Date'].strftime('%Y-%m-%d') if 'Date' in df.columns else 'N/A',
                            'Amount': f"AED {df.loc[idx, 'Debit']:.2f}" if df.loc[idx, 'Debit'] > 0 else f"AED {df.loc[idx, 'Credit']:.2f}"
                        })
                        continue        
        
        # ADDED: Store detailed results
        self.last_ai_stats = ai_stats
        self.ai_processing_details = ai_processing_details
        
        # Add refund detection - ORIGINAL
        df['Is_Refund'] = df['Transaction'].str.lower().str.contains('refund|return|reversal', na=False)
        return df


# ================= Q&A SYSTEM ===================
class EnhancedQASystem:
    """Natural language Q&A system for financial queries"""
    
    def answer(self, question, df):
        """Process natural language questions about financial data"""
        question = question.lower()        
        # Month detection for time-based queries
        month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']        
        asked_month = next((month for month in month_names if month in question), None)        
        if asked_month:
            month_num = {'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12}.get(asked_month, None)            
            if month_num:
                filtered_df = df[df['Date'].dt.month == month_num]
                if filtered_df.empty: return f"No transactions found for {asked_month.title()}"
            else: filtered_df = df
        else: filtered_df = df        
        category_spending = filtered_df.groupby('Category')['Debit'].sum()
        has_balance = filtered_df['Balance'].notna().any()
        month_text = f" in {asked_month.title()}" if asked_month else ""        
        # Income queries
        if any(word in question for word in ['income', 'earned', 'salary received', 'total credit', 'money received']):
            return f"Your total income{month_text}: AED {filtered_df['Credit'].sum():,.2f}"        
        # Balance queries
        if 'balance' in question and has_balance:
            opening, closing = filtered_df['Balance'].dropna().iloc[0], filtered_df['Balance'].dropna().iloc[-1]
            if 'opening' in question: return f"Opening balance: AED {opening:,.2f}"
            elif 'closing' in question: return f"Closing balance: AED {closing:,.2f}"
            else: return f"Opening: AED {opening:,.2f}, Closing: AED {closing:,.2f}"        
        # Category spending queries
        category_map = {'food': 'Food & Dining', 'dining': 'Food & Dining', 'restaurant': 'Food & Dining', 'transport': 'Transportation', 'fuel': 'Transportation', 'taxi': 'Transportation', 'shopping': 'Shopping', 'shop': 'Shopping', 'atm': 'ATM & Cash', 'cash': 'ATM & Cash', 'healthcare': 'Healthcare', 'pharmacy': 'Healthcare', 'medical': 'Healthcare', 'utilities': 'Utilities', 'utility': 'Utilities', 'business': 'Business'}        
        for keyword, category in category_map.items():
            if keyword in question:
                amount = category_spending.get(category, 0)
                return f"You spent AED {amount:,.2f} on {category}{month_text}"        
        # General financial queries
        if 'largest' in question or 'biggest' in question:
            if not category_spending.empty:
                top_cat, top_amount = category_spending.idxmax(), category_spending.max()
                return f"Largest expense{month_text}: {top_cat} with AED {top_amount:,.2f}"
        elif any(word in question for word in ['total expense', 'spent total', 'total spending']): 
            return f"Total expenses{month_text}: AED {filtered_df['Debit'].sum():,.2f}"
        elif 'net' in question or 'cash flow' in question: 
            return f"Net cash flow{month_text}: AED {filtered_df['Net_Amount'].sum():,.2f}"        
        return "Ask about categories (food, transport, etc.), totals, balance, or specify a month (e.g., 'food in August')"

# ================= DASHBOARD FUNCTIONS ===================
def create_interactive_dashboard(df):
    """Create comprehensive interactive financial dashboard"""
    if df.empty: return    
    # Financial summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Transactions", len(df))
    with col2: st.metric("Income", f"AED {df['Credit'].sum():,.2f}")
    with col3: st.metric("Expenses", f"AED {df['Debit'].sum():,.2f}")
    with col4: st.metric("Net Flow", f"AED {df['Net_Amount'].sum():,.2f}")    
    # Balance information if available
    has_balance = df['Balance'].notna().any()
    if has_balance:
        opening, closing = df['Balance'].dropna().iloc[0], df['Balance'].dropna().iloc[-1]
        balance_change = closing - opening
        st.info(f"💰 Balance Change: AED {balance_change:,.2f} (From {opening:,.2f} to {closing:,.2f})")    
    # Monthly analysis
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_summary = df.groupby('Month').agg({'Credit': 'sum', 'Debit': 'sum', 'Balance': 'last', 'Net_Amount': 'sum'}).round(2)    
    # Multi-month interactive charts
    if len(monthly_summary) > 1:
        st.subheader("📈 Monthly Trends")
        st.info("💡 **Zoom & Pan**: Click and drag to zoom, double-click to reset. Each chart can be explored independently.")        
        col1, col2 = st.columns(2)        
        with col1:
            # Monthly balance trend
            if has_balance:
                fig_balance = go.Figure()
                fig_balance.add_trace(go.Scatter(x=[str(month) for month in monthly_summary.index], y=monthly_summary['Balance'], mode='lines+markers', name='Account Balance', line=dict(color='#667eea', width=3), marker=dict(size=8)))
                fig_balance.update_layout(title="Monthly Account Balance", xaxis_title="Month", yaxis_title="Balance (AED)", height=400, showlegend=False, hovermode='x unified')
                st.plotly_chart(fig_balance, use_container_width=True)            
            # Net cash flow chart
            fig_cashflow = go.Figure()
            colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in monthly_summary['Net_Amount']]
            fig_cashflow.add_trace(go.Bar(x=[str(month) for month in monthly_summary.index], y=monthly_summary['Net_Amount'], marker_color=colors, name='Net Cash Flow'))
            fig_cashflow.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            fig_cashflow.update_layout(title="Monthly Net Cash Flow", xaxis_title="Month", yaxis_title="Net Amount (AED)", height=400, showlegend=False, hovermode='x unified')
            st.plotly_chart(fig_cashflow, use_container_width=True)        
        with col2:
            # Income vs expenses comparison
            fig_income = go.Figure()
            fig_income.add_trace(go.Bar(x=[str(month) for month in monthly_summary.index], y=monthly_summary['Credit'], name='Income', marker_color='#4ECDC4'))
            fig_income.add_trace(go.Bar(x=[str(month) for month in monthly_summary.index], y=monthly_summary['Debit'], name='Expenses', marker_color='#FF6B6B'))
            fig_income.update_layout(title="Monthly Income vs Expenses", xaxis_title="Month", yaxis_title="Amount (AED)", height=400, barmode='group', hovermode='x unified')
            st.plotly_chart(fig_income, use_container_width=True)            
            # Category spending trends over time
            category_trends = df.groupby(['Month', 'Category'])['Debit'].sum().unstack(fill_value=0)
            if not category_trends.empty:
                fig_category = go.Figure()
                colors_category = px.colors.qualitative.Set3
                for i, category in enumerate(category_trends.columns):
                    if category_trends[category].sum() > 0:
                        fig_category.add_trace(go.Scatter(x=[str(month) for month in category_trends.index], y=category_trends[category], mode='lines+markers', name=category, stackgroup='one', line=dict(color=colors_category[i % len(colors_category)])))                
                fig_category.update_layout(title="Monthly Spending by Category", xaxis_title="Month", yaxis_title="Amount (AED)", height=400, hovermode='x unified')
                st.plotly_chart(fig_category, use_container_width=True)        
        # Monthly summary table
        st.subheader("📊 Monthly Summary")
        summary_display = monthly_summary.copy()
        for col in ['Credit', 'Debit', 'Balance', 'Net_Amount']:
            if col in summary_display.columns: summary_display[col] = summary_display[col].apply(lambda x: f"AED {x:,.2f}" if pd.notna(x) else "")
        st.dataframe(summary_display, use_container_width=True)        
    else:
        # Single month charts
        col1, col2 = st.columns(2)        
        with col1:
            # Category spending breakdown
            spending = df[df['Debit'] > 0].groupby('Category')['Debit'].sum().sort_values(ascending=False)
            if not spending.empty:
                st.subheader("💰 Spending by Category")
                fig_spending = px.bar(x=spending.values, y=spending.index, orientation='h', title="Spending by Category", labels={'x': 'Amount (AED)', 'y': 'Category'}, color=spending.values, color_continuous_scale='Viridis')
                fig_spending.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_spending, use_container_width=True)        
        with col2:
            # Balance trend over time
            if has_balance and df['Balance'].notna().sum() > 1:
                st.subheader("📈 Balance Trend")
                balance_data = df[df['Balance'].notna()].sort_values('Date')
                fig_balance_single = go.Figure()
                fig_balance_single.add_trace(go.Scatter(x=balance_data['Date'], y=balance_data['Balance'], mode='lines+markers', name='Account Balance', line=dict(color='#667eea', width=3), marker=dict(size=6)))
                fig_balance_single.update_layout(title="Account Balance Over Time", xaxis_title="Date", yaxis_title="Balance (AED)", height=500, showlegend=False, hovermode='x unified')
                st.plotly_chart(fig_balance_single, use_container_width=True)

# ================= PDF REPORT GENERATOR ===================
def generate_enhanced_report_with_beautiful_pdf(df):
    """Generate comprehensive financial report with professional PDF"""
    if df.empty: return "No data available.", None    
    # Calculate key metrics
    category_spending = df.groupby('Category')['Debit'].sum().sort_values(ascending=False)
    has_balance = df['Balance'].notna().any()
    bank_name = df['Bank'].iloc[0] if 'Bank' in df.columns else 'Unknown'    
    total_income, total_expenses, net_flow = df['Credit'].sum(), df['Debit'].sum(), df['Net_Amount'].sum()    
    largest_expense = df.loc[df['Debit'].idxmax()] if df['Debit'].max() > 0 else None
    most_frequent_category = df['Category'].mode().iloc[0] if not df['Category'].mode().empty else 'N/A'
    avg_transaction_size = (total_income + total_expenses) / len(df) if len(df) > 0 else 0    
    # Monthly analysis for multi-month data
    df_temp = df.copy()
    df_temp['Month'] = df_temp['Date'].dt.to_period('M')
    monthly_summary = df_temp.groupby('Month').agg({'Credit': 'sum', 'Debit': 'sum', 'Net_Amount': 'sum'}).round(2)    
    best_month = monthly_summary['Net_Amount'].idxmax() if not monthly_summary.empty else 'N/A'
    worst_month = monthly_summary['Net_Amount'].idxmin() if not monthly_summary.empty else 'N/A'    
    
    # Generate markdown report
    report = f"""# 📊 Comprehensive Financial Analysis Report

**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Bank:** {bank_name}  
**Analysis Period:** {df['Date'].min().strftime('%B %d, %Y')} to {df['Date'].max().strftime('%B %d, %Y')}  
**Total Transactions Analyzed:** {len(df)}

## 💰 Financial Summary
| Metric | Amount (AED) |
|--------|-------------|
| **Total Income** | {total_income:,.2f} |
| **Total Expenses** | {total_expenses:,.2f} |
| **Net Cash Flow** | {net_flow:,.2f} |"""

    # Add balance information if available
    if has_balance:
        opening = df['Balance'].dropna().iloc[0] if not df['Balance'].dropna().empty else 0
        closing = df['Balance'].dropna().iloc[-1] if not df['Balance'].dropna().empty else 0
        account_growth = closing - opening
        report += f"""
| **Opening Balance** | {opening:,.2f} |
| **Closing Balance** | {closing:,.2f} |
| **Account Growth** | {account_growth:,.2f} |"""

    # Category analysis
    report += f"""

## 🏷️ Detailed Category Analysis"""    
    for category, amount in category_spending.items():
        if amount > 0:
            percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            avg_per_transaction = amount / df[df['Category'] == category]['Debit'].count() if df[df['Category'] == category]['Debit'].count() > 0 else 0
            report += f"\n- **{category}:** AED {amount:,.2f} ({percentage:.1f}%) - Avg per transaction: AED {avg_per_transaction:.2f}"    
    
    # Monthly performance analysis
    if len(monthly_summary) > 1:
        report += f"""

## 📅 Monthly Performance Analysis
| Month | Income (AED) | Expenses (AED) | Net Flow (AED) | Performance |
|-------|-------------|---------------|---------------|-------------|"""
        for month, data in monthly_summary.iterrows():
            performance = "🟢 Positive" if data['Net_Amount'] > 0 else "🔴 Negative"
            report += f"\n| {month} | {data['Credit']:,.2f} | {data['Debit']:,.2f} | {data['Net_Amount']:,.2f} | {performance} |"    
    
    # Advanced insights
    report += f"""

## 📈 Advanced Financial Insights
- **Average Transaction Size:** AED {avg_transaction_size:,.2f}
- **Most Active Category:** {most_frequent_category}
- **Financial Health Score:** {'🟢 Excellent' if net_flow > total_expenses * 0.2 else '🟡 Good' if net_flow > 0 else '🔴 Needs Attention'}"""

    if largest_expense is not None:
        report += f"\n- **Largest Single Expense:** AED {largest_expense['Debit']:,.2f} on {largest_expense['Date'].strftime('%B %d, %Y')} ({largest_expense['Transaction'][:50]}...)"    
    if len(monthly_summary) > 1:
        report += f"\n- **Best Performing Month:** {best_month} (AED {monthly_summary.loc[best_month, 'Net_Amount']:,.2f} net flow)"
        report += f"\n- **Month Needing Attention:** {worst_month} (AED {monthly_summary.loc[worst_month, 'Net_Amount']:,.2f} net flow)"

    # Financial metrics
    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0
    report += f"\n- **Savings Rate:** {savings_rate:.1f}% of income"    
    refund_count = df['Is_Refund'].sum()
    if refund_count > 0:
        refund_amount = df[df['Is_Refund']]['Credit'].sum()
        report += f"\n- **Refunds Received:** {refund_count} transactions worth AED {refund_amount:,.2f}"

    # Generate professional PDF report
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm)
        styles = getSampleStyleSheet()
        story = []        
        # Custom styling for professional look
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#2E86AB'), spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica-Bold')
        header_style = ParagraphStyle('CustomHeader', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#A23B72'), spaceBefore=15, spaceAfter=10, fontName='Helvetica-Bold')        
        # Title and header section
        story.append(Paragraph("💰 COMPREHENSIVE FINANCIAL ANALYSIS", title_style))
        story.append(Paragraph(f"{bank_name} Statement Report", styles['Heading3']))
        story.append(Spacer(1, 20))        
        # Report information table
        info_data = [['Report Generated', datetime.now().strftime('%B %d, %Y at %I:%M %p')], ['Analysis Period', f"{df['Date'].min().strftime('%B %d, %Y')} to {df['Date'].max().strftime('%B %d, %Y')}"], ['Total Transactions', f"{len(df)} transactions analyzed"], ['Bank Institution', bank_name]]        
        info_table = Table(info_data, colWidths=[4*cm, 8*cm])
        info_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')), ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('GRID', (0, 0), (-1, -1), 1, colors.grey), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(info_table)
        story.append(Spacer(1, 20))        
        # Financial summary table
        story.append(Paragraph("💰 FINANCIAL SUMMARY", header_style))
        financial_data = [['Metric', 'Amount (AED)', 'Status'], ['Total Income', f"{total_income:,.2f}", '🟢'], ['Total Expenses', f"{total_expenses:,.2f}", '🔴'], ['Net Cash Flow', f"{net_flow:,.2f}", '🟢' if net_flow > 0 else '🔴']]        
        if has_balance:
            opening = df['Balance'].dropna().iloc[0] if not df['Balance'].dropna().empty else 0
            closing = df['Balance'].dropna().iloc[-1] if not df['Balance'].dropna().empty else 0
            account_growth = closing - opening
            financial_data.extend([['Opening Balance', f"{opening:,.2f}", '💼'], ['Closing Balance', f"{closing:,.2f}", '💼'], ['Account Growth', f"{account_growth:,.2f}", '🟢' if account_growth > 0 else '🔴']])        
        financial_table = Table(financial_data, colWidths=[6*cm, 4*cm, 2*cm])
        financial_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 12), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke), ('ALTERNATEROWBACKGROUND', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]), ('GRID', (0, 0), (-1, -1), 1, colors.grey), ('FONTSIZE', (0, 1), (-1, -1), 11)]))
        story.append(financial_table)
        story.append(Spacer(1, 20))        
        # Category analysis table
        story.append(Paragraph("🏷️ CATEGORY ANALYSIS", header_style))
        category_data = [['Category', 'Amount (AED)', 'Percentage', 'Avg/Transaction']]        
        for category, amount in category_spending.items():
            if amount > 0:
                percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
                avg_per_transaction = amount / df[df['Category'] == category]['Debit'].count() if df[df['Category'] == category]['Debit'].count() > 0 else 0
                category_data.append([category, f"{amount:,.2f}", f"{percentage:.1f}%", f"{avg_per_transaction:.2f}"])        
        category_table = Table(category_data, colWidths=[4*cm, 3*cm, 2*cm, 3*cm])
        category_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 11), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke), ('ALTERNATEROWBACKGROUND', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]), ('GRID', (0, 0), (-1, -1), 1, colors.grey), ('FONTSIZE', (0, 1), (-1, -1), 10)]))
        story.append(category_table)
        story.append(Spacer(1, 20))        
        # Monthly performance table (if multi-month data)
        if len(monthly_summary) > 1:
            story.append(Paragraph("📅 MONTHLY PERFORMANCE", header_style))
            monthly_data = [['Month', 'Income (AED)', 'Expenses (AED)', 'Net Flow (AED)', 'Performance']]            
            for month, data in monthly_summary.iterrows():
                performance = "🟢 Positive" if data['Net_Amount'] > 0 else "🔴 Negative"
                monthly_data.append([str(month), f"{data['Credit']:,.2f}", f"{data['Debit']:,.2f}", f"{data['Net_Amount']:,.2f}", performance])            
            monthly_table = Table(monthly_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2*cm])
            monthly_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28A745')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 10), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke), ('ALTERNATEROWBACKGROUND', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]), ('GRID', (0, 0), (-1, -1), 1, colors.grey), ('FONTSIZE', (0, 1), (-1, -1), 9)]))
            story.append(monthly_table)
            story.append(Spacer(1, 20))        
        # Key insights and recommendations
        story.append(Paragraph("📈 KEY INSIGHTS & RECOMMENDATIONS", header_style))        
        insights = [f"💡 Average Transaction Size: AED {avg_transaction_size:,.2f}", f"🎯 Most Active Category: {most_frequent_category}", f"💵 Savings Rate: {savings_rate:.1f}% of total income"]        
        if largest_expense is not None:
            insights.append(f"⚠️ Largest Expense: AED {largest_expense['Debit']:,.2f} on {largest_expense['Date'].strftime('%B %d, %Y')}")        
        if len(monthly_summary) > 1:
            insights.append(f"🏆 Best Month: {best_month} (AED {monthly_summary.loc[best_month, 'Net_Amount']:,.2f} net)")
            insights.append(f"⚠️ Watch Month: {worst_month} (AED {monthly_summary.loc[worst_month, 'Net_Amount']:,.2f} net)")        
        # Financial health assessment with actionable recommendations
        if savings_rate > 20:
            insights.append("🟢 Excellent savings rate - keep up the great work!")
        elif savings_rate > 10:
            insights.append("🟡 Good savings rate - consider increasing if possible")
        elif savings_rate > 0:
            insights.append("🟠 Low savings rate - review expenses for optimization")
        else:
            insights.append("🔴 Negative savings rate - immediate budget review recommended")        
        for insight in insights:
            story.append(Paragraph(insight, styles['Normal']))
            story.append(Spacer(1, 8))        
        story.append(Spacer(1, 20))        
        # Professional footer
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=10, textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph("Generated by Smart Expense Analyser with BART-Large-MNLI AI Classification", footer_style))
        story.append(Paragraph(f"Report ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}", footer_style))        
        # Build and return PDF
        doc.build(story)
        buffer.seek(0)
        pdf_data = buffer.getvalue()        
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        pdf_data = None    
    return report, pdf_data

# ================= STREAMLIT UI ===================
def main():
    """Main Streamlit application with complete user interface"""
    st.set_page_config(page_title="Smart Expense Analyser", page_icon="💰", layout="wide")
    
    # Custom header with gradient styling
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%); border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">💰 Smart Expense Analyser</h1>
        <p style="color: white; margin: 10px 0 0 0;">AI-powered financial analyser for UAE banks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How-it-works section
    with st.expander("💡 How it Works", expanded=False):
        st.markdown("""
        **✨ What this app does:**
        - 📄 **Upload your bank statement** - Supports major UAE banks (FAB, ENBD, CBD, EIB, Mashreq, DIB, ADCB)
        - 🤖 **Smart categorization with AI** - Automatic expense classification
        - 📊 **Interactive insights** - Zoomable charts, monthly trends, balance tracking
        - 💬 **Ask questions** - "What was my total income in August?"
        - 📥 **Professional reports** - Get detailed PDF reports with insights
        
        **🔒 Privacy:** Your data is processed locally and not stored anywhere.
        """)

    # Settings sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        max_file_size = st.slider("Max file size (MB)", 1, 100, 50)
        max_pages = st.slider("Max pages", 1, 30, 20)
        st.info(f"💡 For best results, use monthly statements under {max_file_size}MB")

    # File upload section
    uploaded_file = st.file_uploader("📁 Upload Bank Statement (PDF)", type=['pdf'], help=f"Maximum file size: {max_file_size}MB (adjustable in settings)")

    # Processing section
    if uploaded_file:
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size/(1024*1024):.1f} MB)")
        
        if st.button("🚀 Analyze Report", type="primary"):
            with st.spinner("🔄 Loading BART-Large-MNLI and processing your statement..."):
                try:
                    # Load AI classifier
                    classifier = load_bart_classifier()
                    # Parse statement
                    parser = ComprehensiveUAEParser(uploaded_file.read(), max_pages, max_file_size)
                    df = parser.parse()
                    
                    if df.empty:
                        st.error("❌ No transactions found")
                        return
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return
            
            # Count transactions needing AI and setup progress tracking
            categorizer = SmartCategorizer(classifier)
            temp_df = df.copy()
            def get_category(transaction):
                trans_lower = str(transaction).lower()
                for category, keywords in categorizer.categories.items():
                    if any(keyword in trans_lower for keyword in keywords): return category
                return 'Other'
            temp_df['Category'] = temp_df['Transaction'].apply(get_category)
            ai_needed = (temp_df['Category'] == 'Other').sum()
            
            # Progress tracking setup
            if ai_needed > 0:
                st.info(f"🤖 Processing {ai_needed} transactions with AI...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"🤖 Processing transaction {current}/{total} with AI... ({progress:.0%})")
                
                try:
                    # Apply smart categorization with progress
                    df = categorizer.categorize(df, progress_callback=update_progress)
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("✅ AI categorization complete!")
                    
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store AI stats in session state so they persist
                    if hasattr(categorizer, 'last_ai_stats'):
                        st.session_state['ai_stats'] = categorizer.last_ai_stats
                    if hasattr(categorizer, 'ai_processing_details'):
                        st.session_state['ai_details'] = categorizer.ai_processing_details
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error during AI categorization: {e}")
                    return
            else:
                # No AI needed
                df = categorizer.categorize(df)
                st.info("🎯 All transactions were categorized using rule-based matching!")
                # Store empty AI stats for consistency
                st.session_state['ai_stats'] = {'total_processed': 0, 'successfully_categorized': 0, 'remained_other': 0}
                st.session_state['ai_details'] = []
            
            # Store in session state for reuse
            st.session_state['df'] = df
            st.session_state['qa'] = EnhancedQASystem()
            st.success("✅ Analysis complete!")

    # Results display section
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # AI Processing Summary
        if 'ai_stats' in st.session_state and st.session_state['ai_stats']['total_processed'] > 0:
            st.header("🤖 AI Processing Summary")
            
            stats = st.session_state['ai_stats']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total AI Processed", stats['total_processed'])
            with col2:
                st.metric("✅ Successfully Categorized", stats['successfully_categorized'])
            with col3:
                st.metric("❓ Remained Other", stats['remained_other'])
            with col4:
                if stats['total_processed'] > 0:
                    success_rate = (stats['successfully_categorized'] / stats['total_processed']) * 100
                    st.metric("🎯 AI Success Rate", f"{success_rate:.1f}%")
            
            # Detailed AI processing table
            if 'ai_details' in st.session_state and st.session_state['ai_details']:
                st.subheader("📋 Detailed AI Processing Results")
                
                details_df = pd.DataFrame(st.session_state['ai_details'])
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox(
                        "Filter by Status", 
                        ["All", "✅ Success", "❌ Failed", "⚠️ Error"],
                        key="ai_status_filter"
                    )
                with col2:
                    category_filter = st.selectbox(
                        "Filter by AI Prediction",
                        ["All"] + sorted(details_df['AI Prediction'].unique().tolist()),
                        key="ai_category_filter"
                    )
                
                # Apply filters
                filtered_details = details_df.copy()
                if status_filter != "All":
                    filtered_details = filtered_details[filtered_details['Status'] == status_filter]
                if category_filter != "All":
                    filtered_details = filtered_details[filtered_details['AI Prediction'] == category_filter]
                
                # Display the table
                st.dataframe(
                    filtered_details,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Transaction": st.column_config.TextColumn("Transaction Description", width="large"),
                        "AI Prediction": st.column_config.TextColumn("AI Predicted Category", width="medium"),
                        "Confidence": st.column_config.TextColumn("Confidence Score", width="small"),
                        "Status": st.column_config.TextColumn("Result", width="small"),
                        "Final Category": st.column_config.TextColumn("Assigned Category", width="medium"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Amount": st.column_config.TextColumn("Amount", width="small")
                    }
                )
                
                st.info(f"Showing {len(filtered_details)} of {len(details_df)} AI-processed transactions")
        
        # Interactive dashboard
        st.header("📊 Interactive Financial Dashboard")
        create_interactive_dashboard(df)
        
        # Transaction details table
        st.header("📋 Transaction Details")
        display_df = df.copy()
        display_df['Refund'] = display_df['Is_Refund'].map({True: '✅', False: ''})
        cols = ['Date', 'Transaction', 'Category', 'Debit', 'Credit', 'Net_Amount']
        if df['Balance'].notna().any(): cols.append('Balance')
        cols.append('Refund')
        
        # Format currency columns
        for col in ['Debit', 'Credit', 'Net_Amount', 'Balance']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        
        st.dataframe(display_df[cols], use_container_width=True)
        
        # Report generation section
        st.header("📄 Generate Report")
        if st.button("📋 Generate Report", type="primary"):
            with st.spinner("📊 Creating your comprehensive report..."):
                report, pdf_data = generate_enhanced_report_with_beautiful_pdf(df)
                st.session_state['report'] = report
                st.session_state['pdf_data'] = pdf_data
                st.success("✅ Report generated!")
        
        # Display generated report
        if 'report' in st.session_state:
            st.markdown(st.session_state['report'])
            
            # PDF download button
            if 'pdf_data' in st.session_state and st.session_state['pdf_data']:
                bank_name = df['Bank'].iloc[0].replace(' ', '_') if 'Bank' in df.columns else 'Bank'
                st.download_button("📥 Download PDF Report", data=st.session_state['pdf_data'], file_name=f"{bank_name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", type="secondary")
        
        # Q&A section
        st.header("💬 Ask Questions")
        if 'qa' in st.session_state:
            st.info("💡 **Try asking:** 'What was my total income in September?' or 'How much did I spend on food?'")
            
            # Question input form
            with st.form("qa_form", clear_on_submit=True):
                question = st.text_input("💭 Ask about your finances:", placeholder="What was my total income in August?")
                if st.form_submit_button("🔍 Get Answer", type="primary") and question:
                    answer = st.session_state['qa'].answer(question, df)
                    st.success(f"**Answer:** {answer}")
            
            # Sample questions for easy access
            st.markdown("**💡 Example Questions:**")
            sample_questions = ["What was my total income in September?", "How much did I spend on food in August?", "What's my largest expense category?", "What's my account balance change?", "How much did I spend on healthcare?"]
            
            cols = st.columns(min(len(sample_questions), 3))
            for i, q in enumerate(sample_questions):
                with cols[i % len(cols)]:
                    if st.button(q, key=f"sample_{i}"):
                        answer = st.session_state['qa'].answer(q, df)
                        st.success(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()