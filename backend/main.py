import sqlite3
import os
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
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

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
GOOGLE_API_KEY = "enter your api key" 
# 如果您的 Key 已經在環境變數或直接寫死，請確認這裡有值
if GOOGLE_API_KEY and "YOUR" not in GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # <--- 關鍵修正：讓 ADK 也能讀到 Key
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
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS AIReports (
        Stock_Id TEXT,
        ReportDate DATE,
        NewsAnalysis TEXT,
        CompetitorAnalysis TEXT,
        PRIMARY KEY (Stock_Id, ReportDate)
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
    (混合版) 優先使用 Yahoo 現成數據 (Info) 填補最新年份，歷史數據維持自算
    """
    print(f"--- [混合版] 正在分析 {stock_id} (優先對齊 Yahoo 現成數據) ---")
    df_income, df_balance, df_cashflow = get_dataframes_from_db(stock_id, conn)

    if df_income is None or df_balance is None or df_cashflow is None:
        return False

    ratios_to_save = []
    all_years = df_income.index.sort_values(ascending=False)
    cursor = conn.cursor()

    # 1. 提取 Yahoo Info (來自 CompanyInfo 表)
    # 我們抓取最新的一筆紀錄，轉成字典方便查詢
    cursor.execute("""
        SELECT DataKey, DataValue FROM CompanyInfo 
        WHERE Stock_Id = ? 
        ORDER BY QueryDate DESC
    """, (stock_id,))
    
    # 建立 Info 字典
    yahoo_info = {row[0]: row[1] for row in cursor.fetchall()}

    # 2. 定義 Yahoo Info 的對應表 (我們名稱 -> Yahoo Key)
    # 這些 Key 對應您截圖中的數據
    yahoo_mapping = {
        'Gross Margin': 'grossMargins',
        'Operating Margin': 'operatingMargins',       #
        'Net Profit Margin': 'profitMargins',         #
        'Return on Equity (ROE)': 'returnOnEquity',   #
        'Debt-to-Equity Ratio': 'debtToEquity',       #
        'Current Ratio': 'currentRatio',              #
        'Revenue Growth': 'revenueGrowth',            #
        'EPS Growth': 'earningsGrowth'
    }

    latest_year = all_years[0] if len(all_years) > 0 else 0

    for year in all_years:
        if year not in df_balance.index: continue

        try:
            # 小工具：從 DataFrame 取值
            def get_val(df, y, item): 
                return df.loc[y, item] if item in df.columns else np.nan

            # 先把基礎數據取出來 (計算公式仍需要用到部分數據)
            revenue = get_val(df_income, year, 'Total Revenue')
            gross_profit = get_val(df_income, year, 'Gross Profit')
            op_income = get_val(df_income, year, 'Operating Income')
            net_income = get_val(df_income, year, 'Net Income')
            total_equity = get_val(df_balance, year, 'Total Equity Gross Minority Interest')
            total_debt = get_val(df_balance, year, 'Total Debt')
            current_assets = get_val(df_balance, year, 'Current Assets')
            current_liabilities = get_val(df_balance, year, 'Current Liabilities')
            invested_capital = get_val(df_balance, year, 'Invested Capital')
            
            # --- [核心邏輯] 定義一個函式來決定用誰的數據 ---
            def get_ratio_value(ratio_name, calculated_value):
                """
                如果:
                1. 現在是最新一年 (year == latest_year)
                2. Yahoo Info 裡面有這個欄位
                3. Yahoo 的值有效
                => 則回傳 Yahoo 的值 (優先權高)
                => 否則回傳 calculated_value (自算值)
                """
                # 只有最新一年才嘗試用 Yahoo Info (因為 Info 是 TTM 或 Current)
                if year == latest_year and ratio_name in yahoo_mapping:
                    y_key = yahoo_mapping[ratio_name]
                    
                    # 檢查 Info 裡有沒有這個值且不是 'None'
                    if y_key in yahoo_info and yahoo_info[y_key] and yahoo_info[y_key] != 'None':
                        try:
                            y_val = float(yahoo_info[y_key])
                            
                            # [特別處理] 單位換算
                            # Yahoo 的 DebtToEquity 是 41.60 (代表 41.6%)，需轉成 0.416
                            if y_key == 'debtToEquity':
                                y_val = y_val / 100
                            
                            # Debug 訊息 (可選)
                            # print(f"   ★ {ratio_name}: 使用 Yahoo 數據 {y_val} 替代自算 {calculated_value}")
                            return y_val
                        except:
                            pass # 轉換失敗就繼續用算的
                
                return calculated_value

            # ==========================================
            # === 1. 獲利能力 ===
            # ==========================================

            # Gross Margin
            if revenue > 0 and not pd.isna(gross_profit):
                calc_val = gross_profit/revenue
                final_val = get_ratio_value('Gross Margin', calc_val)
                ratios_to_save.append((stock_id, year, 'profitability', 'Gross Margin', final_val, 'Hybrid'))

            # Operating Margin
            if revenue > 0 and not pd.isna(op_income):
                calc_val = op_income/revenue
                final_val = get_ratio_value('Operating Margin', calc_val)
                ratios_to_save.append((stock_id, year, 'profitability', 'Operating Margin', final_val, 'Hybrid'))

            # Net Profit Margin
            if revenue > 0 and not pd.isna(net_income):
                calc_val = net_income/revenue
                final_val = get_ratio_value('Net Profit Margin', calc_val)
                ratios_to_save.append((stock_id, year, 'profitability', 'Net Profit Margin', final_val, 'Hybrid'))
            
            # ROE (這就是您提到的 8.15% vs 8.9% 的關鍵修正)
            if total_equity > 0 and not pd.isna(net_income):
                calc_val = net_income/total_equity
                final_val = get_ratio_value('Return on Equity (ROE)', calc_val)
                ratios_to_save.append((stock_id, year, 'profitability', 'Return on Equity (ROE)', final_val, 'Hybrid'))

            # ROIC (Yahoo 通常只有 ROA，ROIC 還是得自算)
            if invested_capital > 0 and not pd.isna(net_income): 
                ratios_to_save.append((stock_id, year, 'profitability', 'ROIC', net_income/invested_capital, 'Net/IC'))

            # ==========================================
            # === 2. 槓桿與流動性 ===
            # ==========================================

            # Debt-to-Equity
            if total_equity > 0 and not pd.isna(total_debt):
                calc_val = total_debt/total_equity
                final_val = get_ratio_value('Debt-to-Equity Ratio', calc_val)
                ratios_to_save.append((stock_id, year, 'leverage', 'Debt-to-Equity Ratio', final_val, 'Hybrid'))
            
            # Current Ratio
            if current_liabilities > 0 and not pd.isna(current_assets):
                calc_val = current_assets/current_liabilities
                final_val = get_ratio_value('Current Ratio', calc_val)
                ratios_to_save.append((stock_id, year, 'leverage', 'Current Ratio', final_val, 'Hybrid'))

            # 利息保障倍數 (Yahoo Info 較少直接提供，維持自算)
            interest_expense = get_val(df_income, year, 'Interest Expense')
            if interest_expense > 0 and not pd.isna(op_income):
                ratios_to_save.append((stock_id, year, 'leverage', 'Interest Coverage Ratio', op_income/interest_expense, 'Op/Int'))

            # Net Debt / EBITDA (維持自算)
            ebitda = get_val(df_income, year, 'EBITDA')
            net_debt = get_val(df_balance, year, 'Net Debt')
            if ebitda > 0 and not pd.isna(net_debt):
                ratios_to_save.append((stock_id, year, 'leverage', 'Net Debt / EBITDA', net_debt/ebitda, 'NetDebt/EBITDA'))

            # ==========================================
            # === 3. 經營效率 (維持自算) ===
            # ==========================================
            # 這些項目 Yahoo Info 比較少直接給，維持自算確保趨勢圖連貫
            
            total_assets = get_val(df_balance, year, 'Total Assets')
            inventory = get_val(df_balance, year, 'Inventory')
            cost_of_revenue = get_val(df_income, year, 'Cost Of Revenue')
            accounts_receivable = get_val(df_balance, year, 'Accounts Receivable')

            if total_assets > 0 and not pd.isna(revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Asset Turnover', revenue/total_assets, 'Rev/Assets'))
            if inventory > 0 and not pd.isna(cost_of_revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Inventory Turnover', cost_of_revenue/inventory, 'Cost/Inv'))
            if accounts_receivable > 0 and not pd.isna(revenue):
                ratios_to_save.append((stock_id, year, 'efficiency', 'Receivables Turnover', revenue/accounts_receivable, 'Rev/AR'))

            # ==========================================
            # === 4. 成長性 (混合) ===
            # ==========================================
            
            prev_year = year - 1
            if prev_year in df_income.index:
                try:
                    prev_revenue = get_val(df_income, prev_year, 'Total Revenue')
                    prev_net_income = get_val(df_income, prev_year, 'Net Income')
                    prev_eps = get_val(df_income, prev_year, 'Basic EPS')
                    basic_eps = get_val(df_income, year, 'Basic EPS')
                    
                    # 營收成長
                    if prev_revenue > 0 and not pd.isna(revenue):
                        calc_val = (revenue - prev_revenue) / prev_revenue
                        # Yahoo 的 revenueGrowth 通常是 Quarterly YoY，可能與年度成長不同
                        # 但如果您希望看到截圖上的 -4.10%，這裡可以開啟混合模式
                        ratios_to_save.append((stock_id, year, 'growth', 'Revenue Growth', calc_val, 'Hybrid'))
                    
                    # 淨利成長
                    if prev_net_income != 0 and not pd.isna(net_income) and not pd.isna(prev_net_income):
                        growth = (net_income - prev_net_income) / abs(prev_net_income)
                        ratios_to_save.append((stock_id, year, 'growth', 'Net Income Growth', growth, '(NI - PrevNI)/abs(PrevNI)'))

                    # EPS 成長
                    if not pd.isna(basic_eps) and not pd.isna(prev_eps) and prev_eps != 0:
                        growth = (basic_eps - prev_eps) / abs(prev_eps)
                        ratios_to_save.append((stock_id, year, 'growth', 'EPS Growth', growth, '(EPS - PrevEPS)/abs(PrevEPS)'))

                    # FCF 成長 (維持自算)
                    if prev_year in df_cashflow.index:
                        prev_fcf = get_val(df_cashflow, prev_year, 'Free Cash Flow')
                        fcf = get_val(df_cashflow, year, 'Free Cash Flow')
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
# 3.5. 同學的 AI 分析模組 (整合區)
# ==========================================

def get_competitor_dataframe_markdown(stock_id):
    """
    功能：抓取目標公司與競爭對手的財務數據，並轉為 Markdown 表格
    """
    try:
        ticker = stock_id
        shell = yf.Ticker(ticker)
        info = shell.info
        
        # 如果抓不到 industryKey，無法比較，回傳 None
        if 'industryKey' not in info:
            return None, None

        # 1. 建立目標公司的數據 List (使用 .get 避免缺漏報錯)
        target_list = [
            ticker, 
            info.get('dividendYield', 0), info.get('trailingPE', 0), info.get('priceToSalesTrailing12Months', 0),
            info.get('profitMargins', 0), info.get('priceToBook', 0), info.get('trailingEps', 0),
            info.get('enterpriseToEbitda', 0), info.get('currentRatio', 0), info.get('debtToEquity', 0),
            info.get('returnOnAssets', 0), info.get('returnOnEquity', 0), info.get('trailingPegRatio', 0)
        ]

        # 2. 找出競爭對手 (取前 4 名)
        industry = yf.Industry(info['industryKey'])
        competitors = list(industry.top_companies.index.values)[:4] 
        
        columns = ['Ticker', 'Dividend Yield', 'Trailing PE', 'TTM PS', 'Profit Margin', 'PB Ratio', 
                   'Trailing EPS', 'EV/EBITDA', 'Current Ratio', 'Debt-to-Equity', 'ROA', 'ROE', 'PEG Ratio']
        
        compare_df = pd.DataFrame([target_list], columns=columns)

        # 3. 抓取競爭者數據
        for comp in competitors:
            try:
                comp_info = yf.Ticker(comp).info
                comp_list = [
                    comp, 
                    comp_info.get('dividendYield', 0), comp_info.get('trailingPE', 0), comp_info.get('priceToSalesTrailing12Months', 0),
                    comp_info.get('profitMargins', 0), comp_info.get('priceToBook', 0), comp_info.get('trailingEps', 0),
                    comp_info.get('enterpriseToEbitda', 0), comp_info.get('currentRatio', 0), comp_info.get('debtToEquity', 0),
                    comp_info.get('returnOnAssets', 0), comp_info.get('returnOnEquity', 0), comp_info.get('trailingPegRatio', 0)
                ]
                compare_df.loc[len(compare_df)] = comp_list
            except Exception as e:
                print(f"Skipping competitor {comp}: {e}")

        # 4. 轉成 Markdown
        compare_df = compare_df.round(4)
        return compare_df.to_markdown(index=False), info.get('longBusinessSummary', '')

    except Exception as e:
        print(f"Error getting competitor data: {e}")
        return None, None

async def run_ai_analysis_agent(stock_id, summary, comparison_markdown):
    """
    功能：執行 Google ADK Agent 進行分析 (修正版：使用 run_debug 並清洗數據)
    """
    # 1. 定義新聞分析 Agent
    print(f"--- AI Agent: Analyzing News for {stock_id} ---")
    google_news_agent = Agent(
        name="GoogleNewsAgent",
        model="gemini-2.5-flash",
        instruction="""You are a specialized news arrange agent. Read the company's summary and use Google Search 
        to find news related to the company. Generate a concise comment highlighting key risks or opportunities.""",
        tools=[google_search],
        output_key="google_news_arrangement", # 結果會存在這裡
    )
    
    news_runner = InMemoryRunner(agent=google_news_agent)
    news_prompt = f"{stock_id} summary: \n\n{summary}"
    
    # 🔴 修正 1: 改回用 run_debug (它支援直接傳字串)
    news_events = await news_runner.run_debug(news_prompt)
    
    # 🔴 修正 2: 清洗數據，從複雜的事件物件中抓出文字
    news_text = "News analysis failed or no output generated."
    try:
        # 遍歷所有事件，尋找包含我們 output_key 的那個事件
        for event in news_events:
            # 檢查 event.actions.state_delta 是否包含我們的結果
            if event.actions and event.actions.state_delta and "google_news_arrangement" in event.actions.state_delta:
                news_text = event.actions.state_delta["google_news_arrangement"]
                break # 找到了就停止
    except Exception as e:
        print(f"Error parsing news output: {e}")
        news_text = str(news_events) # 萬一解析失敗，至少顯示原始資料以便除錯

    # 2. 定義競爭者分析 Agent
    print(f"--- AI Agent: Analyzing Competitors for {stock_id} ---")
    competitors_agent = Agent(
        name="CompetitorsAgent",
        model="gemini-2.5-flash",
        instruction="""Analyze the financial data for the company and its competitors. 
        Compare them and provide insights on valuation (undervalued/overvalued).""",
        output_key="comparing_competitors", # 結果會存在這裡
    )

    comp_runner = InMemoryRunner(agent=competitors_agent)
    comp_prompt = f"Please analyze the following financial data for {stock_id} and its competitors:\n\n{comparison_markdown}"
    
    # 執行與清洗
    comp_events = await comp_runner.run_debug(comp_prompt)
    comp_text = "Competitor analysis failed or no output generated."
    try:
        for event in comp_events:
            if event.actions and event.actions.state_delta and "comparing_competitors" in event.actions.state_delta:
                comp_text = event.actions.state_delta["comparing_competitors"]
                break
    except Exception as e:
        print(f"Error parsing competitor output: {e}")
        comp_text = str(comp_events)

    return str(news_text), str(comp_text)

# ==========================================
# 4. Agent API & Logic
# ==========================================

# ==========================================
# 4. Agent API & Logic (修改版：加入記憶功能)
# ==========================================

@app.on_event("startup")
def startup():
    create_fundamental_tables()

# 🔥 新增：全域變數，用來儲存對話歷史
chat_session = None

def get_chat_session():
    """確保有一個活著的對話 Session"""
    global chat_session
    if chat_session is None:
        # 初始化一個有記憶的模型
        model = genai.GenerativeModel("gemini-2.5-flash")
        chat_session = model.start_chat(history=[])
    return chat_session

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
    
    # 取得全域對話 Session
    session = get_chat_session()

    # --- 邏輯分支 ---
    
    # 情況 A: 使用者只問了普通問題，且沒提到公司，且 Agent 還是張白紙
    if (ticker == "NONE" or " " in ticker or len(ticker) > 10) and not session.history:
        model = genai.GenerativeModel("gemini-2.5-flash")
        reply = model.generate_content(f"User said: '{user_msg}'. Reply politely as a financial assistant asking for a company name.").text
        return {"status": "chat", "message": reply}

    # 情況 B: 使用者在追問 (沒提到新公司，但 Agent 已經有記憶)
    if (ticker == "NONE" or " " in ticker or len(ticker) > 10) and session.history:
        print(f"💬 使用者正在追問: {user_msg}")
        # 直接把問題丟給有記憶的 session
        response = session.send_message(user_msg)
        return {"status": "chat", "message": response.text}

    # 情況 C: 使用者提到了新的公司 (例如 "Analyze AAPL") -> 執行分析並將資料注入記憶
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
        
        # 3. AI 總結 (這一步最關鍵：我們將數據注入到對話 Session 中)
        context = get_context_str(ticker)
        
        # 這裡我們不建立新模型，而是將龐大的數據變成一個 Prompt，傳給有記憶的 Session
        final_prompt = f"""
        [System Update: New Market Data Loaded]
        Target Company: {ticker}
        
        Historical Ratios (2021-2024):
        {context}
        
        Valuation Model Result:
        {dcf_report}
        
        User Question: "{user_msg}"
        
        Instruction: Provide a comprehensive investment analysis. 
        Note: Remember this data for future follow-up questions.
        """
        
        # 傳送給有記憶的 Session
        response = session.send_message(final_prompt)
        
        return {
            "status": "analysis_complete",
            "ticker": ticker,
            "data": data_records,
            "reply": response.text
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}
@app.post("/api/analyze_ai/{stock_id}")
async def analyze_stock_ai(stock_id: str):
    """
    觸發同學寫的 AI 分析功能，並存入資料庫
    """
    stock_id = stock_id.upper()
    conn = get_db_connection()
    
    # 1. 準備數據 (yfinance & Markdown)
    print(f"正在準備 {stock_id} 的 AI 分析數據...")
    comparison_md, summary = get_competitor_dataframe_markdown(stock_id)
    
    if not comparison_md or not summary:
        conn.close()
        return {"status": "error", "message": "無法取得 Yahoo Finance 數據或競爭者資料"}

    # 2. 執行 AI 分析 (這可能會花幾秒鐘)
    try:
        # 呼叫定義好的 async 函式
        news_analysis, competitor_analysis = await run_ai_analysis_agent(stock_id, summary, comparison_md)
    except Exception as e:
        conn.close()
        return {"status": "error", "message": f"AI 模型執行失敗: {str(e)}"}

    # 3. 存入資料庫
    today = dt.date.today().strftime("%Y-%m-%d")
    cursor = conn.cursor()
    
    # 使用 INSERT OR REPLACE 確保同一天重複按按鈕會更新
    cursor.execute('''
        INSERT OR REPLACE INTO AIReports (Stock_Id, ReportDate, NewsAnalysis, CompetitorAnalysis)
        VALUES (?, ?, ?, ?)
    ''', (stock_id, today, news_analysis, competitor_analysis))
    
    conn.commit()
    conn.close()
    
    return {
        "status": "success", 
        "data": {
            "news_analysis": news_analysis,
            "competitor_analysis": competitor_analysis
        }
    }

@app.get("/api/get_ai_report/{stock_id}")
def get_ai_report(stock_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 抓取最新的一份報告
    cursor.execute('''
        SELECT ReportDate, NewsAnalysis, CompetitorAnalysis 
        FROM AIReports 
        WHERE Stock_Id = ? 
        ORDER BY ReportDate DESC 
        LIMIT 1
    ''', (stock_id.upper(),))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "date": row[0],
            "news_analysis": row[1],
            "competitor_analysis": row[2]
        }
    else:
        return {"message": "尚無分析報告"}