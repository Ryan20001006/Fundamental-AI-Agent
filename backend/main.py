import sqlite3
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
import pandas_datareader.data as web
import statsmodels.api as sm
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. 設定與初始化
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "stock.db"

# ⚠️ 請確認 API Key 是否正確
GOOGLE_API_KEY = "AIzaSyA_ITRbmHfB1yT2HBoAa_JK553MlrK1fgk" 
# 如果您的 Key 已經在環境變數或直接寫死，請確認這裡有值
if GOOGLE_API_KEY and "YOUR" not in GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- Pydantic Models (定義請求格式) ---
class StockRequest(BaseModel):
    ticker: str

# 👇 補回這個 class 定義
class ChatRequest(BaseModel):
    message: str

def get_db_connection():
    return sqlite3.connect(DB_NAME)

# ==========================================
# 2. 資料庫與數據處理
# ==========================================

def create_fundamental_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CompanyInfo (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        Stock_Id TEXT, QueryDate DATE, DataKey TEXT, DataValue TEXT,
        UNIQUE(Stock_Id, DataKey, QueryDate)
    );''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS FinancialStatements (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        Stock_Id TEXT, StatementType TEXT, Item TEXT, ReportDate DATE, Value REAL,
        UNIQUE(Stock_Id, StatementType, Item, ReportDate)
    );''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CalculatedRatios (
        sno INTEGER PRIMARY KEY AUTOINCREMENT,
        Stock_Id TEXT, ReportYear INT, Category TEXT, RatioName TEXT, RatioValue REAL, Formula TEXT,
        UNIQUE(Stock_Id, ReportYear, RatioName)
    );''')
    
    conn.commit()
    conn.close()

def download_and_store_fundamentals(stock_id):
    print(f"📥 正在下載 {stock_id} 的數據...")
    conn = get_db_connection()
    try:
        stock = yf.Ticker(stock_id)
        if not stock.info: return False
        
        today = dt.date.today().strftime('%Y-%m-%d')
        cursor = conn.cursor()

        # 1. Info
        info_data = []
        for k, v in stock.info.items():
            info_data.append((stock_id, today, k, str(v)))
        cursor.executemany('INSERT OR IGNORE INTO CompanyInfo (Stock_Id, QueryDate, DataKey, DataValue) VALUES (?, ?, ?, ?)', info_data)

        # 2. Financials (不含 2025 預估)
        statements = {'Income': stock.financials, 'BalanceSheet': stock.balance_sheet, 'CashFlow': stock.cashflow}
        all_stmt_data = []
        
        for stmt_type, df in statements.items():
            if df.empty: continue
            
            df = df.reset_index().melt(id_vars='index', var_name='ReportDate', value_name='Value')
            df.rename(columns={'index': 'Item'}, inplace=True)
            df['ReportDate'] = pd.to_datetime(df['ReportDate']).dt.strftime('%Y-%m-%d')
            df = df.dropna(subset=['Value'])
            
            for row in df.itertuples(index=False):
                all_stmt_data.append((stock_id, stmt_type, row.Item, row.ReportDate, row.Value))
        
        if all_stmt_data:
            cursor.executemany('INSERT OR IGNORE INTO FinancialStatements (Stock_Id, StatementType, Item, ReportDate, Value) VALUES (?, ?, ?, ?, ?)', all_stmt_data)
        
        conn.commit()
        return True
    except Exception as e:
        print(f"下載錯誤: {e}")
        return False
    finally:
        conn.close()

def get_dataframes_from_db(stock_id, conn):
    query = "SELECT StatementType, Item, ReportDate, Value FROM FinancialStatements WHERE Stock_Id = ?"
    df_all = pd.read_sql(query, conn, params=(stock_id,))
    
    if df_all.empty:
        return None, None, None

    def get_pivot(stmt_type):
        d = df_all[df_all['StatementType'] == stmt_type]
        if d.empty: return pd.DataFrame()
        p = d.pivot_table(index='ReportDate', columns='Item', values='Value')
        p.index = pd.to_datetime(p.index).year
        return p.sort_index(ascending=False)

    return get_pivot('Income'), get_pivot('BalanceSheet'), get_pivot('CashFlow')

def calculate_financial_ratios(stock_id, conn):
    """
    (專業完整版) 計算 16 個關鍵財務指標
    """
    print(f"--- [專業版] 正在分析 {stock_id} 的全方位歷史財務比率 ---")
    df_income, df_balance, df_cashflow = get_dataframes_from_db(stock_id, conn)

    if df_income is None or df_balance is None or df_cashflow is None:
        return False

    ratios_to_save = []
    all_years = df_income.index.sort_values(ascending=False)
    cursor = conn.cursor()

    for year in all_years:
        if year not in df_balance.index: continue

        try:
            def get_val(df, y, item): 
                return df.loc[y, item] if item in df.columns else np.nan

            # 損益表
            revenue = get_val(df_income, year, 'Total Revenue')
            gross_profit = get_val(df_income, year, 'Gross Profit')
            op_income = get_val(df_income, year, 'Operating Income')
            net_income = get_val(df_income, year, 'Net Income')
            cost_of_revenue = get_val(df_income, year, 'Cost Of Revenue')
            interest_expense = get_val(df_income, year, 'Interest Expense')
            ebitda = get_val(df_income, year, 'EBITDA')
            basic_eps = get_val(df_income, year, 'Basic EPS')

            # 資產負債表
            total_assets = get_val(df_balance, year, 'Total Assets')
            total_equity = get_val(df_balance, year, 'Total Equity Gross Minority Interest')
            total_debt = get_val(df_balance, year, 'Total Debt')
            net_debt = get_val(df_balance, year, 'Net Debt')
            inventory = get_val(df_balance, year, 'Inventory')
            accounts_receivable = get_val(df_balance, year, 'Accounts Receivable')
            current_assets = get_val(df_balance, year, 'Current Assets')
            current_liabilities = get_val(df_balance, year, 'Current Liabilities')
            invested_capital = get_val(df_balance, year, 'Invested Capital')

            # 現金流量表
            ocf = np.nan
            fcf = np.nan
            if year in df_cashflow.index:
                ocf = get_val(df_cashflow, year, 'Operating Cash Flow')
                fcf = get_val(df_cashflow, year, 'Free Cash Flow')

            # === 1. 獲利能力 ===
            if revenue > 0 and not pd.isna(gross_profit):
                ratios_to_save.append((stock_id, year, 'profitability', 'Gross Margin', gross_profit/revenue, 'Gross/Rev'))
            if revenue > 0 and not pd.isna(op_income):
                ratios_to_save.append((stock_id, year, 'profitability', 'Operating Margin', op_income/revenue, 'Op/Rev'))
            if revenue > 0 and not pd.isna(net_income):
                ratios_to_save.append((stock_id, year, 'profitability', 'Net Profit Margin', net_income/revenue, 'Net/Rev'))
            if total_equity > 0 and not pd.isna(net_income):
                ratios_to_save.append((stock_id, year, 'profitability', 'Return on Equity (ROE)', net_income/total_equity, 'Net/Equity'))
            if invested_capital > 0 and not pd.isna(net_income): 
                ratios_to_save.append((stock_id, year, 'profitability', 'ROIC', net_income/invested_capital, 'Net/IC'))

            # === 2. 槓桿與流動性 ===
            if total_equity > 0 and not pd.isna(total_debt):
                ratios_to_save.append((stock_id, year, 'leverage', 'Debt-to-Equity Ratio', total_debt/total_equity, 'Debt/Equity'))
            if interest_expense > 0 and not pd.isna(op_income):
                ratios_to_save.append((stock_id, year, 'leverage', 'Interest Coverage Ratio', op_income/interest_expense, 'Op/Int'))
            if current_liabilities > 0 and not pd.isna(current_assets):
                ratios_to_save.append((stock_id, year, 'leverage', 'Current Ratio', current_assets/current_liabilities, 'CA/CL'))
            if ebitda > 0 and not pd.isna(net_debt):
                ratios_to_save.append((stock_id, year, 'leverage', 'Net Debt / EBITDA', net_debt/ebitda, 'NetDebt/EBITDA'))

            # === 3. 經營效率 ===
            if total_assets > 0 and not pd.isna(revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Asset Turnover', revenue/total_assets, 'Rev/Assets'))
            if inventory > 0 and not pd.isna(cost_of_revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Inventory Turnover', cost_of_revenue/inventory, 'Cost/Inv'))
            if accounts_receivable > 0 and not pd.isna(revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Receivables Turnover', revenue/accounts_receivable, 'Rev/AR'))

            # === 4. 成長性 ===
            prev_year = year - 1
            if prev_year in df_income.index:
                try:
                    prev_revenue = get_val(df_income, prev_year, 'Total Revenue')
                    prev_net_income = get_val(df_income, prev_year, 'Net Income')
                    prev_eps = get_val(df_income, prev_year, 'Basic EPS')

                    if prev_revenue > 0 and not pd.isna(revenue):
                        growth = (revenue - prev_revenue) / prev_revenue
                        ratios_to_save.append((stock_id, year, 'growth', 'Revenue Growth', growth, '(Rev - PrevRev)/PrevRev'))
                    
                    if prev_net_income != 0 and not pd.isna(net_income) and not pd.isna(prev_net_income):
                        growth = (net_income - prev_net_income) / abs(prev_net_income)
                        ratios_to_save.append((stock_id, year, 'growth', 'Net Income Growth', growth, '(NI - PrevNI)/abs(PrevNI)'))

                    if not pd.isna(basic_eps) and not pd.isna(prev_eps) and prev_eps != 0:
                        growth = (basic_eps - prev_eps) / abs(prev_eps)
                        ratios_to_save.append((stock_id, year, 'growth', 'EPS Growth', growth, '(EPS - PrevEPS)/abs(PrevEPS)'))

                    if prev_year in df_cashflow.index:
                        prev_fcf = get_val(df_cashflow, prev_year, 'Free Cash Flow')
                        if not pd.isna(fcf) and not pd.isna(prev_fcf) and prev_fcf != 0:
                            growth = (fcf - prev_fcf) / abs(prev_fcf)
                            ratios_to_save.append((stock_id, year, 'growth', 'FCF Growth', growth, '(FCF - PrevFCF)/abs(PrevFCF)'))
                except KeyError: pass

        except KeyError:
            continue

    if ratios_to_save:
        cursor.executemany('''
        INSERT OR IGNORE INTO CalculatedRatios
            (Stock_Id, ReportYear, Category, RatioName, RatioValue, Formula)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ratios_to_save)
        conn.commit()
        return True
    return False

def get_context_str(stock_id):
    conn = get_db_connection()
    try:
        df = pd.read_sql("SELECT ReportYear, RatioName, RatioValue FROM CalculatedRatios WHERE Stock_Id = ? ORDER BY ReportYear DESC, RatioName", conn, params=(stock_id,))
        if df.empty: return "No Data"
        df_pivot = df.pivot_table(index='RatioName', columns='ReportYear', values='RatioValue')
        return df_pivot.to_markdown()
    finally:
        conn.close()

# ==========================================
# 3. 進階估值模型 (整合區)
# ==========================================

def calculate_fama_french_coe(ticker_symbol, lookback_years=5):
    """計算 Fama-French 權益成本"""
    print(f"📊 [模型 1/3] 計算 Fama-French CoE ({ticker_symbol})...")
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=lookback_years*365)
    
    try:
        ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)[0]
        ff_data = ff_data / 100
        ff_data.index = ff_data.index.to_timestamp()
    except Exception:
        print("⚠️ Fama-French 數據獲取失敗，使用 Fallback 10%")
        return 0.10 

    ticker = yf.Ticker(ticker_symbol)
    stock = ticker.history(start=start_date, end=end_date, interval='1mo')
    if stock.empty: return 0.10

    stock_returns = stock['Close'].pct_change().dropna()
    stock_returns.index = stock_returns.index.to_period('M')
    ff_data.index = ff_data.index.to_period('M')
    
    data = pd.merge(stock_returns, ff_data, left_index=True, right_index=True)
    data.columns = ['Stock_Return', 'Mkt-RF', 'SMB', 'HML', 'RF']
    data['Excess_Return'] = data['Stock_Return'] - data['RF']
    
    X = sm.add_constant(data[['Mkt-RF', 'SMB', 'HML']])
    model = sm.OLS(data['Excess_Return'], X).fit()
    
    exp_mkt = data['Mkt-RF'].mean() * 12
    exp_smb = data['SMB'].mean() * 12
    exp_hml = data['HML'].mean() * 12
    rf = data['RF'].iloc[-1] * 12
    
    coe = rf + (model.params['Mkt-RF'] * exp_mkt) + (model.params['SMB'] * exp_smb) + (model.params['HML'] * exp_hml)
    return coe

def project_fcf_from_eps_filtered(ticker_symbol):
    """使用 Forward EPS 預測 FCF"""
    print(f"🔮 [模型 2/3] 預測 FCF ({ticker_symbol})...")
    stock = yf.Ticker(ticker_symbol)
    
    def filter_post_2020(df):
        df.columns = pd.to_datetime(df.columns)
        return df[[c for c in df.columns if c.year >= 2021]]

    try:
        financials = filter_post_2020(stock.financials)
        cashflow = filter_post_2020(stock.cashflow)
        
        net_income = financials.loc['Net Income']
        fcf = cashflow.loc['Operating Cash Flow'] - abs(cashflow.loc['Capital Expenditure'])
        
        ratios = (fcf / net_income).replace([np.inf, -np.inf], np.nan).dropna()
        avg_ratio = ratios.mean() if not ratios.empty else 1.0
        
        forward_eps = stock.info.get('forwardEps') or stock.info.get('trailingEps')
        return forward_eps * avg_ratio
    except Exception:
        return 0

def calculate_dcf(ticker_symbol, coe, fcfps_FTM, projection_years=5, terminal_growth_rate=0.00):
    """執行 DCF 估值 (0% 成長率)"""
    print(f"💰 [模型 3/3] 執行最終 DCF 估值...")
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    current_price = info.get('currentPrice')
    currency = info.get('currency', 'USD')
    if currency == 'GBp':
        current_price = current_price / 100
        currency = 'GBP'
        
    shares = info.get('sharesOutstanding')
    financials = stock.financials
    balance = stock.balance_sheet
    
    try:
        int_exp = abs(financials.loc['Interest Expense'].iloc[0]) if 'Interest Expense' in financials.index else 0
        debt = balance.loc['Total Debt'].iloc[0] if 'Total Debt' in balance.index else 0
        tax_rate = 0.21
        cost_debt = (int_exp / debt) * (1 - tax_rate) if debt > 0 else 0.05
        
        mkt_cap = shares * current_price
        total_val = mkt_cap + debt
        wacc = ((mkt_cap/total_val) * coe) + ((debt/total_val) * cost_debt)
    except:
        wacc = coe
    
    growth_rate_projection = 0.00 # 0% 成長
    future_fcf = [fcfps_FTM * shares * ((1 + growth_rate_projection) ** i) for i in range(1, projection_years + 1)]
    
    term_val = (future_fcf[-1] * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    disc_fcfs = sum([f / ((1 + wacc) ** (i + 1)) for i, f in enumerate(future_fcf)])
    disc_tv = term_val / ((1 + wacc) ** projection_years)
    
    intrinsic_val = (disc_fcfs + disc_tv) / shares
    status = "低估 (Undervalued)" if intrinsic_val > current_price else "高估 (Overvalued)"
    
    return f"""
    [Advanced DCF Valuation]
    - Ticker: {ticker_symbol}
    - Current Price: {current_price:.2f} {currency}
    - Fair Value: {intrinsic_val:.2f} {currency}
    - Conclusion: {status}
    --------------------------------
    - WACC: {wacc:.2%}
    - Proj. FCF/Share: {fcfps_FTM:.2f}
    - Growth Assumption: 0.0%
    """

def run_advanced_valuation(ticker):
    """總指揮函式"""
    coe = calculate_fama_french_coe(ticker) or 0.10
    fcf_ftm = project_fcf_from_eps_filtered(ticker)
    if fcf_ftm <= 0: return "Error: Insufficient Data for Valuation"
    return calculate_dcf(ticker, coe, fcf_ftm)

# ==========================================
# 4. Agent API & Logic
# ==========================================

@app.on_event("startup")
def startup():
    create_fundamental_tables()

def extract_ticker_from_text(text: str):
    """Agent 耳朵：增強版意圖識別"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    Role: Financial Extraction Engine
    Task: Extract ticker from input. Support Chinese.
    Input: "{text}"
    Output: ONLY the ticker (e.g. SHEL.L, AAPL). If none, output NONE.
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return "NONE"

@app.post("/api/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    if not download_and_store_fundamentals(ticker):
        raise HTTPException(status_code=404, detail="Download failed")
    conn = get_db_connection()
    calculate_financial_ratios(ticker, conn)
    df = pd.read_sql("SELECT * FROM CalculatedRatios WHERE Stock_Id = ?", conn, params=(ticker,))
    conn.close()
    return {"status": "success", "data": df.to_dict(orient="records")}

@app.post("/api/agent-chat")
def agent_chat(req: ChatRequest):
    user_msg = req.message
    ticker = extract_ticker_from_text(user_msg)
    
    if ticker == "NONE" or " " in ticker or len(ticker) > 10:
        model = genai.GenerativeModel("gemini-2.5-pro")
        reply = model.generate_content(f"User said: '{user_msg}'. Reply politely as a financial assistant asking for a company name.").text
        return {"status": "chat", "message": reply}
    
    try:
        # 1. 執行基礎分析 (過去)
        download_and_store_fundamentals(ticker)
        conn = get_db_connection()
        calculate_financial_ratios(ticker, conn)
        
        df = pd.read_sql("SELECT * FROM CalculatedRatios WHERE Stock_Id = ?", conn, params=(ticker,))
        conn.close()
        data_records = df.to_dict(orient="records")
        
        # 2. 執行進階估值 (未來)
        dcf_report = run_advanced_valuation(ticker)
        
        # 3. AI 總結
        context = get_context_str(ticker)
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        final_prompt = f"""
        User Question: "{user_msg}"
        
        Historical Ratios (2021-2024):
        {context}
        
        Valuation Model Result:
        {dcf_report}
        
        Please provide a comprehensive investment analysis answering the user's question.
        Combine the historical health with the valuation result.
        """
        final_response = model.generate_content(final_prompt)
        
        return {
            "status": "analysis_complete",
            "ticker": ticker,
            "data": data_records,
            "reply": final_response.text
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}