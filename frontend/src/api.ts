import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000/api';

export interface FinancialRatio {
  Stock_Id: string;
  ReportYear: number;
  Category: string;
  RatioName: string;
  RatioValue: number;
}

export interface AIReport {
  date: string;
  news_analysis: string;
  competitor_analysis: string;
}

// 與 Agent 對話
export const talkToAgent = async (message: string) => {
  const response = await axios.post(`${API_URL}/agent-chat`, { message });
  return response.data;
};

// 觸發 AI 進階分析 (新聞 + 競爭者)
export const triggerAIAnalysis = async (ticker: string) => {
  const response = await axios.post(`${API_URL}/analyze_ai/${ticker}`);
  return response.data;
};

// 獲取 AI 分析報告
export const getAIReport = async (ticker: string): Promise<AIReport | null> => {
  try {
    const response = await axios.get(`${API_URL}/get_ai_report/${ticker}`);
    if (response.data.message) return null; // 尚無報告
    return response.data;
  } catch (error) {
    console.error("Error fetching AI report:", error);
    return null;
  }
};