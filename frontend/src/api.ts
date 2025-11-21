import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000/api';

export interface FinancialRatio {
  Stock_Id: string;
  ReportYear: number;
  Category: string;
  RatioName: string;
  RatioValue: number;
}

// 這是原本的分析 (可以留著備用，但 Agent 模式主要用下面那個)
export const analyzeStock = async (ticker: string): Promise<FinancialRatio[]> => {
  const response = await axios.post(`${API_URL}/analyze`, { ticker });
  return response.data.data;
};

export const generateMemo = async (ticker: string): Promise<string> => {
  const response = await axios.post(`${API_URL}/generate-memo`, { ticker });
  return response.data.memo;
};

// ✅ 新增：與 Agent 對話的專用函式
export const talkToAgent = async (message: string) => {
  const response = await axios.post(`${API_URL}/agent-chat`, { message });
  return response.data;
};