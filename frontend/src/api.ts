import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000/api';

export interface FinancialRatio {
  Stock_Id: string;
  ReportYear: number;
  Category: string;
  RatioName: string;
  RatioValue: number;
}

export const analyzeStock = async (ticker: string): Promise<FinancialRatio[]> => {
  const response = await axios.post(`${API_URL}/analyze`, { ticker });
  return response.data.data;
};

export const generateMemo = async (ticker: string): Promise<string> => {
  const response = await axios.post(`${API_URL}/generate-memo`, { ticker });
  return response.data.memo;
};