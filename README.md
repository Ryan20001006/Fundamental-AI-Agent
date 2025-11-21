# Fundamental AI Agent

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)


**Fundamental AI Agent** is a full-stack financial analysis tool designed to automate the workflow of a fund manager. It fetches real-time stock data, calculates key financial ratios (Profitability, Leverage, Efficiency, Growth), and leverages Google Gemini AI to generate qualitative investment memos.

![Dashboard Screenshot](./screenshot.png)

---

## Key Features

* [cite_start]**Automated Data Extraction**: Instantly fetches Balance Sheet, Income Statement, and Cash Flow data using `yfinance`.
* **Financial Modeling**: Automatically calculates financial ratios including:
    * [cite_start]**Profitability**: ROE, Net Profit Margin, Gross Margin.
    * [cite_start]**Leverage**: Debt-to-Equity Ratio.
    * [cite_start]**Growth**: EPS Growth.
* [cite_start]**AI Investment Memo**: Generates a professional investment thesis and risk assessment using **Google Gemini 2.5 Flash**.
* ** Interactive Dashboard**: Visualizes financial trends over time using **Recharts**.
* [cite_start]**Local Caching**: Uses **SQLite** to store fetched data, reducing API calls and improving performance.

---

##  Tech Stack

### **Frontend**
* **Framework**: React (Vite)
* **Language**: TypeScript
* **Visualization**: Recharts
* **HTTP Client**: Axios
* **Styling**: CSS3 (Custom Responsive Design)

### **Backend**
* [cite_start]**Framework**: FastAPI 
* [cite_start]**Database**: SQLite 
* [cite_start]**AI Model**: Google Generative AI (Gemini) 
* [cite_start]**Data Processing**: Pandas, NumPy, Yfinance 

---

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites
* Node.js (v16+)
* Python (v3.9+)
* Google AI Studio API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/RYAN20001006/Fundamental-AI-Agent.git](https://github.com/RYAN20001006/Fundamental-AI-Agent.git)
cd Fundamental-AI-Agent
```
2. Backend Setup
Navigate to the backend folder and set up the Python environment.

```Bash

cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
# venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn yfinance pandas numpy google-generativeai tabulate
```
⚠️ Important: Open backend/main.py  and replace the GOOGLE_API_KEY variable with your actual API Key.

3. Frontend Setup
Open a new terminal, navigate to the frontend folder, and install dependencies.

```Bash

cd frontend
npm install
```
How to Run
To run the application, you must start both the backend server and the frontend client simultaneously in separate terminals.

Terminal 1: Backend
```Bash

cd backend
source venv/bin/activate
uvicorn main:app --reload
```
The API will run at http://127.0.0.1:8000

Terminal 2: Frontend
```Bash

cd frontend
npm run dev
```

The application will run at http://localhost:5173

Project Structure
```Plaintext

Fundamental-AI-Agent/
├── backend/
│   ├── main.py             # FastAPI Backend & Logic 
│   ├── stock.db            # SQLite Database
│   └── venv/               # Virtual Environment
│
├── frontend/
│   ├── src/
│   │   ├── api.ts          # API Integration
│   │   ├── App.tsx         # Main Dashboard Component
│   │   ├── App.css         # Styles
│   │   └── main.tsx        # React Entry Point [cite: 34]
│   └── package.json        # Frontend Dependencies
│
└── README.md
```
