import { useState, useRef, useEffect } from 'react';
import { talkToAgent, triggerAIAnalysis, getAIReport, type FinancialRatio, type AIReport } from './api';
import ReactMarkdown from 'react-markdown';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Send, Bot, User, Loader2, Sparkles, FileText } from 'lucide-react';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false); // 控制 AI 分析按鈕的讀取狀態
  
  const [messages, setMessages] = useState<{role: 'user' | 'agent', content: string}[]>([
    {role: 'agent', content: 'I am the fundamental AI Chatbot. Ask me anything about a company! (e.g., "Analyze Shell for me")'}
  ]);

  const [chartData, setChartData] = useState<FinancialRatio[]>([]);
  const [currentTicker, setCurrentTicker] = useState('');
  const [aiReport, setAiReport] = useState<AIReport | null>(null); // 儲存 AI 報告
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  // 自動捲動到最新訊息
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 當 Ticker 改變時，嘗試抓取舊的 AI 報告 (如果有的話)
  useEffect(() => {
    if (currentTicker) {
      setAiReport(null); // 切換公司先清空舊報告
      getAIReport(currentTicker).then(report => {
        if (report) setAiReport(report);
      });
    }
  }, [currentTicker]);

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
        
        // 自動嘗試抓取該公司的 AI 報告
        const report = await getAIReport(result.ticker);
        setAiReport(report);

      } else {
        setMessages(prev => [...prev, { role: 'agent', content: result.message }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: 'Error connecting to the server.' }]);
    } finally {
      setLoading(false);
    }
  };

  // 觸發 AI 深度分析 (整合同學的部分)
  const handleRunDeepAnalysis = async () => {
    if (!currentTicker) return;
    setAiLoading(true);
    try {
      // 1. 呼叫後端執行分析
      const res = await triggerAIAnalysis(currentTicker);
      if (res.status === 'success') {
        // 2. 執行成功後，將結果存入狀態顯示
        setAiReport({
          date: new Date().toISOString().split('T')[0],
          news_analysis: res.data.news_analysis,
          competitor_analysis: res.data.competitor_analysis
        });
        setMessages(prev => [...prev, { role: 'agent', content: `✅ Deep analysis for **${currentTicker}** is ready! Check the dashboard.` }]);
      } else {
        alert("Analysis failed: " + res.message);
      }
    } catch (e) {
      alert("Error triggering analysis");
    } finally {
      setAiLoading(false);
    }
  };

  const getChartData = (name: string) => {
    return chartData.filter(d => d.RatioName === name)
               .sort((a, b) => a.ReportYear - b.ReportYear)
               .map(d => ({ year: d.ReportYear, value: d.RatioValue }));
  };

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
              placeholder="Type your question (e.g. Analyze SHEL.L)..."
              disabled={loading}
            />
            <button onClick={handleSend} disabled={loading}>
              <Send size={18} />
            </button>
          </div>
        </div>

        {/* 右側：儀表板 (有數據時才出現) */}
        {chartData.length > 0 && (
          <div className="dashboard-section">
            
            <div className="dashboard-header" style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
              <h2>📊 {currentTicker} Analysis</h2>
              
              {/* 新增：執行深度分析的按鈕 */}
              <button 
                onClick={handleRunDeepAnalysis} 
                disabled={aiLoading}
                className="ai-btn"
                style={{
                  backgroundColor: aiLoading ? '#94a3b8' : '#7c3aed', 
                  color: 'white',
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: 'none',
                  cursor: aiLoading ? 'not-allowed' : 'pointer'
                }}
              >
                {aiLoading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
                {aiLoading ? 'Analyzing...' : 'Run Deep AI Analysis'}
              </button>
            </div>

            {/* 新增：顯示 AI 報告的區塊 */}
            {aiReport && (
              <div className="ai-report-container" style={{background: '#fff', padding: '20px', borderRadius: '12px', marginBottom: '20px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}>
                <h3 style={{display:'flex', alignItems:'center', gap:'8px', color:'#4f46e5'}}>
                  <FileText size={20} /> 
                  AI Integrated Report ({aiReport.date})
                </h3>
                
                <div className="report-section">
                  <h4 style={{color:'#0f172a', borderBottom:'2px solid #e2e8f0', paddingBottom:'8px'}}>📰 Market News Analysis</h4>
                  <div style={{fontSize:'0.95rem', lineHeight:'1.6', color:'#334155'}}>
                    <ReactMarkdown>{aiReport.news_analysis}</ReactMarkdown>
                  </div>
                </div>

                <div className="report-section" style={{marginTop:'20px'}}>
                  <h4 style={{color:'#0f172a', borderBottom:'2px solid #e2e8f0', paddingBottom:'8px'}}>⚔️ Competitor Comparison</h4>
                  <div style={{fontSize:'0.95rem', lineHeight:'1.6', color:'#334155'}}>
                    <ReactMarkdown>{aiReport.competitor_analysis}</ReactMarkdown>
                  </div>
                </div>
              </div>
            )}

            {/* 原有的圖表 */}
            <div className="charts-grid">
              <RenderChart title="ROE (Return on Equity)" ratioName="Return on Equity (ROE)" color="#8884d8" />
              <RenderChart title="Gross Margin" ratioName="Gross Margin" color="#ffc658" />
              <RenderChart title="Revenue Growth" ratioName="Revenue Growth" color="#0088FE" />
              <RenderChart title="Debt-to-Equity Ratio" ratioName="Debt-to-Equity Ratio" color="#FFBB28" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
