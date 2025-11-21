import sqlite3
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. 設定與初始化
# ==========================================

app = FastAPI()

# --- 設定 CORS (解決 405/Network Error 的關鍵) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有前端網址連線
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有動作 (GET, POST, OPTIONS)
    allow_headers=["*"],
)

DB_NAME = "stock.db"

GOOGLE_API_KEY = "USE YOUR OWN API KEY"
genai.configure(api_key=GOOGLE_API_KEY)

class StockRequest(BaseModel):
    ticker: str

def get_db_connection():
    return sqlite3.connect(DB_NAME)

# ==========================================
# 2. Database
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
    print(f"Downloading {stock_id} Data...")
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

        # 2. Financials
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
        print(f"Download Error: {e}")
        return False
    finally:
        conn.close()

# --- Helper Function ---
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
    This is the AI Financial Ratio Calculator
    """
    print(f"--- Analysing {stock_id} Historical Financial Data ---")

    # 1. Collect raw data
    df_income, df_balance, df_cashflow = get_dataframes_from_db(stock_id, conn)

    if df_income is None or df_balance is None or df_cashflow is None:
        return {"error": f"unable to extract {stock_id} 's financial data"}

    ratios_to_save = []
    all_years = df_income.index.sort_values(ascending=False)
    cursor = conn.cursor() 

    for year in all_years:
        if year not in df_balance.index:
            continue

        # ---------------------------------------------------------
        # A. Data Extraction
        # ---------------------------------------------------------
        try:
            # --- Income Statement ---
            revenue = df_income.loc[year, 'Total Revenue'] if 'Total Revenue' in df_income.columns else np.nan
            gross_profit = df_income.loc[year, 'Gross Profit'] if 'Gross Profit' in df_income.columns else np.nan
            op_income = df_income.loc[year, 'Operating Income'] if 'Operating Income' in df_income.columns else np.nan
            net_income = df_income.loc[year, 'Net Income'] if 'Net Income' in df_income.columns else np.nan
            cost_of_revenue = df_income.loc[year, 'Cost Of Revenue'] if 'Cost Of Revenue' in df_income.columns else np.nan
            interest_expense = df_income.loc[year, 'Interest Expense'] if 'Interest Expense' in df_income.columns else np.nan
            ebitda = df_income.loc[year, 'EBITDA'] if 'EBITDA' in df_income.columns else np.nan
            basic_eps = df_income.loc[year, 'Basic EPS'] if 'Basic EPS' in df_income.columns else np.nan

            # --- Balance Sheet ---
            total_assets = df_balance.loc[year, 'Total Assets'] if 'Total Assets' in df_balance.columns else np.nan
            total_equity = df_balance.loc[year, 'Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in df_balance.columns else np.nan
            total_debt = df_balance.loc[year, 'Total Debt'] if 'Total Debt' in df_balance.columns else np.nan
            net_debt = df_balance.loc[year, 'Net Debt'] if 'Net Debt' in df_balance.columns else np.nan
            inventory = df_balance.loc[year, 'Inventory'] if 'Inventory' in df_balance.columns else np.nan
            accounts_receivable = df_balance.loc[year, 'Accounts Receivable'] if 'Accounts Receivable' in df_balance.columns else np.nan
            current_assets = df_balance.loc[year, 'Current Assets'] if 'Current Assets' in df_balance.columns else np.nan
            current_liabilities = df_balance.loc[year, 'Current Liabilities'] if 'Current Liabilities' in df_balance.columns else np.nan
            invested_capital = df_balance.loc[year, 'Invested Capital'] if 'Invested Capital' in df_balance.columns else np.nan

            # --- Cashflow ---
            ocf = df_cashflow.loc[year, 'Operating Cash Flow'] if (year in df_cashflow.index and 'Operating Cash Flow' in df_cashflow.columns) else np.nan
            fcf = df_cashflow.loc[year, 'Free Cash Flow'] if (year in df_cashflow.index and 'Free Cash Flow' in df_cashflow.columns) else np.nan

        except KeyError as e:
            print(f"Year {year} error: {e}")
            continue

        # ---------------------------------------------------------
        # B. Ratio Calculation
        # ---------------------------------------------------------

        # === 1. Profitability  ===
        if revenue > 0 and not pd.isna(gross_profit):
            val = gross_profit / revenue
            ratios_to_save.append((stock_id, year, 'profitability', 'Gross Margin', val, 'Gross Profit / Revenue'))

        if revenue > 0 and not pd.isna(op_income):
            val = op_income / revenue
            ratios_to_save.append((stock_id, year, 'profitability', 'Operating Margin', val, 'Operating Income / Revenue'))

        if revenue > 0 and not pd.isna(net_income):
            val = net_income / revenue
            ratios_to_save.append((stock_id, year, 'profitability', 'Net Profit Margin', val, 'Net Income / Revenue'))

        if total_equity > 0 and not pd.isna(net_income):
            val = net_income / total_equity
            ratios_to_save.append((stock_id, year, 'profitability', 'Return on Equity (ROE)', val, 'Net Income / Total Equity'))

        if invested_capital > 0 and not pd.isna(net_income):
            val = net_income / invested_capital
            ratios_to_save.append((stock_id, year, 'profitability', 'ROIC', val, 'Net Income / Invested Capital'))

        # === 2. Leverage & Liquidity  ===
        if total_equity > 0 and not pd.isna(total_debt):
            val = total_debt / total_equity
            ratios_to_save.append((stock_id, year, 'leverage', 'Debt-to-Equity Ratio', val, 'Total Debt / Total Equity'))

        if interest_expense > 0 and not pd.isna(op_income):
            val = op_income / interest_expense
            ratios_to_save.append((stock_id, year, 'leverage', 'Interest Coverage Ratio', val, 'Operating Income / Interest Expense'))

        if current_liabilities > 0 and not pd.isna(current_assets):
            val = current_assets / current_liabilities
            ratios_to_save.append((stock_id, year, 'leverage', 'Current Ratio', val, 'Current Assets / Current Liabilities'))

        if ebitda > 0 and not pd.isna(net_debt):
            val = net_debt / ebitda
            ratios_to_save.append((stock_id, year, 'leverage', 'Net Debt / EBITDA', val, 'Net Debt / EBITDA'))

        # === 3. Efficiency  ===
        if total_assets > 0 and not pd.isna(revenue):
            val = revenue / total_assets
            ratios_to_save.append((stock_id, year, 'efficiency', 'Asset Turnover', val, 'Revenue / Total Assets'))

        if inventory > 0 and not pd.isna(cost_of_revenue):
            val = cost_of_revenue / inventory
            ratios_to_save.append((stock_id, year, 'efficiency', 'Inventory Turnover', val, 'Cost of Revenue / Inventory'))

        if accounts_receivable > 0 and not pd.isna(revenue):
            val = revenue / accounts_receivable
            ratios_to_save.append((stock_id, year, 'efficiency', 'Receivables Turnover', val, 'Revenue / Accounts Receivable'))

        # === 4. Growth  ===
        prev_year = year - 1
        if prev_year in df_income.index:
            try:
                prev_revenue = df_income.loc[prev_year, 'Total Revenue']
                prev_net_income = df_income.loc[prev_year, 'Net Income']
                prev_eps = df_income.loc[prev_year, 'Basic EPS'] if 'Basic EPS' in df_income.columns else np.nan

                if prev_revenue > 0 and not pd.isna(revenue):
                    growth = (revenue - prev_revenue) / prev_revenue
                    ratios_to_save.append((stock_id, year, 'growth', 'Revenue Growth', growth, '(Rev - PrevRev)/PrevRev'))

                if prev_net_income != 0 and not pd.isna(net_income) and not pd.isna(prev_net_income):
                    growth = (net_income - prev_net_income) / abs(prev_net_income)
                    ratios_to_save.append((stock_id, year, 'growth', 'Net Income Growth', growth, '(NI - PrevNI)/abs(PrevNI)'))

                if not pd.isna(basic_eps) and not pd.isna(prev_eps) and prev_eps != 0:
                    growth = (basic_eps - prev_eps) / abs(prev_eps)
                    ratios_to_save.append((stock_id, year, 'growth', 'EPS Growth', growth, '(EPS - PrevEPS)/abs(PrevEPS)'))

                if prev_year in df_cashflow.index and 'Free Cash Flow' in df_cashflow.columns:
                    prev_fcf = df_cashflow.loc[prev_year, 'Free Cash Flow']
                    if not pd.isna(fcf) and not pd.isna(prev_fcf) and prev_fcf != 0:
                        growth = (fcf - prev_fcf) / abs(prev_fcf)
                        ratios_to_save.append((stock_id, year, 'growth', 'FCF Growth', growth, '(FCF - PrevFCF)/abs(PrevFCF)'))

            except KeyError:
                pass

    # C. Save the results
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
        df = pd.read_sql("SELECT ReportYear, RatioName, RatioValue FROM CalculatedRatios WHERE Stock_Id = ? ORDER BY ReportYear DESC", conn, params=(stock_id,))
        if df.empty: return "No Data"
        df_pivot = df.pivot_table(index='RatioName', columns='ReportYear', values='RatioValue')
        return df_pivot.to_markdown()
    finally:
        conn.close()
# ==========================================
# 3. API 
# ==========================================

@app.on_event("startup")
def startup():
    create_fundamental_tables()

@app.post("/api/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    
    if not download_and_store_fundamentals(ticker):
        raise HTTPException(status_code=404, detail="Download failed")
    
    conn = get_db_connection()
    try:
        calculate_financial_ratios(ticker, conn)
    finally:
        conn.close()
    
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM CalculatedRatios WHERE Stock_Id = ?", conn, params=(ticker,))
    conn.close()
    
    return {"status": "success", "data": df.to_dict(orient="records")}

@app.post("/api/generate-memo")
def memo(req: StockRequest):
    ticker = req.ticker.upper()
    context = get_context_str(ticker)
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Act as a fund manager. Write a short investment memo for {ticker} based on this data:\n{context}"
    
    try:
        res = model.generate_content(prompt)
        return {"memo": res.text}
    except Exception as e:
        return {"memo": f"Error generating memo: {str(e)}"}
    
#4. AI Agent

class ChatRequest(BaseModel):
    message: str

def extract_ticker_from_text(text: str):
    """
    This is where the AI Agent will determine if the data needs to be downloaded.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are a financial AI agent. Extract the stock ticker symbol from the user's query.
    If the user mentions a company name (e.g., "Apple", "Shell", "TSMC"), convert it to the correct Yahoo Finance ticker (e.g., AAPL, SHEL.L, 2330.TW).
    
    User Query: "{text}"
    
    Output ONLY the ticker string (e.g., AAPL). 
    If no company is mentioned or the intent is unclear, output "NONE".
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "NONE"

@app.post("/api/agent-chat")
def agent_chat(req: ChatRequest):
    user_msg = req.message
    print(f"Agent message received: {user_msg}")
    
    # 步驟 A: 思考 (意圖識別)
    ticker = extract_ticker_from_text(user_msg)
    print(f"Ticker: {ticker}")
    
    # 如果 AI 聽不懂或是使用者只是在閒聊
    if ticker == "NONE":
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat_reply = model.generate_content(f"You are a helpful financial assistant. The user said: '{user_msg}'. Respond politely that you can help analyze stocks if they provide a company name.").text
        return {
            "status": "chat", 
            "message": chat_reply
        }
    
    # 步驟 B: 行動 (使用既有的分析工具)
    try:
        # 1. 呼叫我們原本寫好的下載函式
        download_success = download_and_store_fundamentals(ticker)
        if not download_success:
             return {"status": "chat", "message": f"抱歉，我找到了代碼 {ticker}，但無法從資料源獲取它的詳細數據。"}

        # 2. 呼叫原本的計算函式
        conn = get_db_connection()
        calculate_financial_ratios(ticker, conn)
        
        # 3. 撈取數據回傳給前端畫圖
        df = pd.read_sql("SELECT * FROM CalculatedRatios WHERE Stock_Id = ?", conn, params=(ticker,))
        conn.close()
        
        data_records = df.to_dict(orient="records")
        
        # 4. 生成最終的 Agent 回覆 (基於數據)
        context = get_context_str(ticker)
        model = genai.GenerativeModel("gemini-2.5-flash") # 或用 gemini-2.5-pro
        
        final_prompt = f"""
        User Question: '{user_msg}'
        
        Based on the following financial data for {ticker}:
        {context}
        
        Please provide a concise answer to the user's question and a brief investment summary.
        Use Markdown format for bolding key numbers.
        """
        final_response = model.generate_content(final_prompt)
        
        return {
            "status": "analysis_complete",
            "ticker": ticker,
            "data": data_records,
            "reply": final_response.text
        }
        
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"status": "error", "message": f"分析過程中發生錯誤: {str(e)}"}