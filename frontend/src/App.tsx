import { useState } from 'react';
import { analyzeStock, generateMemo, type FinancialRatio } from './api';
import ReactMarkdown from 'react-markdown';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { FileText } from 'lucide-react';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState<FinancialRatio[]>([]);
  const [memo, setMemo] = useState('');
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  const handleSearch = async () => {
    if (!ticker) return;
    setLoading(true); setMsg('Analysing...'); setData([]); setMemo('');
    
    try {
      const result = await analyzeStock(ticker);
      setData(result);
      setMsg('Forming investment memo...');
      const memoRes = await generateMemo(ticker);
      setMemo(memoRes);
      setMsg('Done！');
    } catch (error) {
      console.error(error);
      alert('Error, an error has occurred');
    } finally {
      setLoading(false);
    }
  };

  const getChartData = (name: string) => {
    return data.filter(d => d.RatioName === name)
               .sort((a, b) => a.ReportYear - b.ReportYear)
               .map(d => ({ year: d.ReportYear, value: d.RatioValue }));
  };

  return (
    <div className="container">
      <h1>Fundamental AI Agent</h1>
      <div className="search-box">
        <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="Please type in the ticker you want to analyze (e.g. AAPL)" />
        <button onClick={handleSearch} disabled={loading}>{loading ? 'Process' : 'Analyze'}</button>
      </div>
      <p>{msg}</p>

      {data.length > 0 && (
        <div className="dashboard">
          {/* 左邊：圖表區 (這裡我們改成直列顯示多張圖) */}
          <div className="charts-column">
            
            {/* 圖表 1: ROE */}
            <div className="chart">
              <h3>ROE </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getChartData('ROE')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* 圖表 2: 淨利率 (Net Profit Margin) */}
            <div className="chart">
              <h3>Net Profit Margin </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getChartData('Net Profit Margin')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#82ca9d" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* 圖表 3: 負債權益比 (Debt-to-Equity) */}
            <div className="chart">
              <h3>Debt-to-Equity </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getChartData('Debt-to-Equity Ratio')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#ff7300" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart">
              <h3>EPS Growth </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={getChartData('EPS Growth')}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#ff7300" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

          </div>

          {/* 右邊：AI 報告區 */}
          <div className="memo">
            <h3><FileText /> Investment Memo</h3>
            <div className="markdown"><ReactMarkdown>{memo}</ReactMarkdown></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;