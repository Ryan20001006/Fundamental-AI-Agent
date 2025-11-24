import { useState, useRef, useEffect } from 'react';
import { talkToAgent, type FinancialRatio } from './api';
import ReactMarkdown from 'react-markdown';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  const [messages, setMessages] = useState<{role: 'user' | 'agent', content: string}[]>([
    {role: 'agent', content: 'I am the fundamental AI Chatbot. Ask me anything about a company! (e.g., "Analyze Apple for me")'}
  ]);

  const [chartData, setChartData] = useState<FinancialRatio[]>([]);
  const [currentTicker, setCurrentTicker] = useState('');
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = input;
    setInput(''); 
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    try {
      const result = await talkToAgent(userMsg);

      if (result.status === 'analysis_complete') {
        setChartData(result.data);
        setCurrentTicker(result.ticker);
        setMessages(prev => [...prev, { role: 'agent', content: result.reply }]);
      } else {
        setMessages(prev => [...prev, { role: 'agent', content: result.message }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: 'Error connecting to the server. Please check if the backend is running.' }]);
    } finally {
      setLoading(false);
    }
  };

  const getChartData = (name: string) => {
    return chartData.filter(d => d.RatioName === name)
               .sort((a, b) => a.ReportYear - b.ReportYear)
               .map(d => ({ year: d.ReportYear, value: d.RatioValue }));
  };

  // 抽取出畫圖的組件，讓程式碼更乾淨
  const RenderChart = ({ title, ratioName, color }: { title: string, ratioName: string, color: string }) => (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={getChartData(ratioName)}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="year" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

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
                  <Loader2 className="spin" size={16} /> Analyzing...
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
              placeholder="Type your question..."
              disabled={loading}
            />
            <button onClick={handleSend} disabled={loading}>
              <Send size={18} />
            </button>
          </div>
        </div>

        {/* 右側：分析儀表板 (顯示更多圖表！) */}
        {chartData.length > 0 && (
          <div className="dashboard-section">
            <div className="dashboard-header">
              <h2>📊 {currentTicker} Analysis</h2>
            </div>
            <div className="charts-grid">
              
              {/* 1. 獲利能力 (Profitability) */}
              <RenderChart title="ROE (Return on Equity)" ratioName="Return on Equity (ROE)" color="#8884d8" />
              <RenderChart title="ROIC (Return on Invested Capital)" ratioName="ROIC" color="#82ca9d" />
              <RenderChart title="Gross Margin" ratioName="Gross Margin" color="#ffc658" />
              <RenderChart title="Net Profit Margin" ratioName="Net Profit Margin" color="#ff7300" />

              {/* 2. 成長性 (Growth) */}
              <RenderChart title="Revenue Growth" ratioName="Revenue Growth" color="#0088FE" />
              <RenderChart title="EPS Growth" ratioName="EPS Growth" color="#00C49F" />

              {/* 3. 槓桿與效率 (Leverage & Efficiency) */}
              <RenderChart title="Debt-to-Equity Ratio" ratioName="Debt-to-Equity Ratio" color="#FFBB28" />
              <RenderChart title="Inventory Turnover" ratioName="Inventory Turnover" color="#FF8042" />
              
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;