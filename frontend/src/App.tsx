import { useState, useRef, useEffect } from 'react';
import { talkToAgent, type FinancialRatio } from './api';
import ReactMarkdown from 'react-markdown';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // 對話紀錄：role 區分是誰說話，content 是內容
  const [messages, setMessages] = useState<{role: 'user' | 'agent', content: string}[]>([
    {role: 'agent', content: 'I am the fundamental AI Chatbot, please type in your questions.'}
  ]);

  // 儲存分析後的數據 (用來畫圖)
  const [chartData, setChartData] = useState<FinancialRatio[]>([]);
  const [currentTicker, setCurrentTicker] = useState('');
  
  // 自動捲動到最新訊息
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = input;
    setInput(''); // 清空輸入框
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      // 呼叫 Agent
      const result = await talkToAgent(userMsg);

      if (result.status === 'analysis_complete') {
        // 1. 如果 Agent 決定進行分析，它會回傳數據和 ticker
        setChartData(result.data);
        setCurrentTicker(result.ticker);
        // 2. 顯示 Agent 的分析結論
        setMessages(prev => [...prev, { role: 'agent', content: result.reply }]);
      } else {
        // 3. 如果只是閒聊，或者沒找到股票
        setMessages(prev => [...prev, { role: 'agent', content: result.message }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: '抱歉，連線發生錯誤，請確認後端是否開啟。' }]);
    } finally {
      setLoading(false);
    }
  };

  const getChartData = (name: string) => {
    return chartData.filter(d => d.RatioName === name)
               .sort((a, b) => a.ReportYear - b.ReportYear)
               .map(d => ({ year: d.ReportYear, value: d.RatioValue }));
  };

  return (
    <div className="container">
      <h1>Fundamental AI Agent</h1>
      
      <div className="main-layout">
        {/* 左側：聊天室 */}
        <div className="chat-section">
          <div className="chat-window">
            {messages.map((msg, index) => (
              <div key={index} className={`message-row ${msg.role}`}>
                <div className="avatar">
                  {msg.role === 'agent' ? <Bot size={20} /> : <User size={20} />}
                </div>
                <div className={`bubble ${msg.role}`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            ))}
            {loading && (
              <div className="message-row agent">
                <div className="avatar"><Bot size={20} /></div>
                <div className="bubble agent flex-center">
                  <Loader2 className="spin" size={16} /> Analysing...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="input-area">
            <input 
              value={input} 
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Type in your question..."
              disabled={loading}
            />
            <button onClick={handleSend} disabled={loading}>
              <Send size={18} />
            </button>
          </div>
        </div>

        {/* 右側：分析儀表板 (有數據時才顯示) */}
        {chartData.length > 0 && (
          <div className="dashboard-section">
            <div className="dashboard-header">
              <h2>📊 {currentTicker} Analysis</h2>
            </div>
            <div className="charts-grid">
              <div className="chart-card">
                <h3>ROE</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={getChartData('ROE')}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3>Net Profit Margin </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={getChartData('Net Profit Margin')}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#82ca9d" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3>Debt-to-Equity Ratio</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={getChartData('Debt-to-Equity Ratio')}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="value" stroke="#ff7300" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3>EPS Growth</h3>
                <ResponsiveContainer width="100%" height={200}>
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
          </div>
        )}
      </div>
    </div>
  );
}

export default App;