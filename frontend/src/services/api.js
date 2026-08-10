import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// attach auth token to all requests - nst
api.interceptors.request.use((config) => {
  const auth = JSON.parse(localStorage.getItem('dermavision-auth') || '{}');
  // zustand persists stuff in auth.state
  const token = auth.state?.token || auth.token;
  
  // skip for demo/local tokens (those are fake)
  if (token && !token.startsWith('local-token') && !token.startsWith('demo-token')) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// handle auth failures
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const auth = JSON.parse(localStorage.getItem('dermavision-auth') || '{}');
      const token = auth.state?.token || auth.token;
      
      if (token && (token.startsWith('local-token') || token.startsWith('demo-token'))) {
        // this is fine, we're running in local mode
        console.log('Using local authentication, API features unavailable');
      }
    }
    return Promise.reject(error);
  }
);

// all our API endpoints grouped nicely - 
export const authAPI = {
  login: (email, password) => api.post('/api/v1/auth/login', { email, password }),
  register: (data) => api.post('/api/v1/auth/register', data),
  logout: () => api.post('/api/v1/auth/logout'),
  forgotPassword: (email) => api.post('/api/v1/auth/forgot-password', { email }),
  resetPassword: (token, password) => api.post('/api/v1/auth/reset-password', { token, password }),
  getProfile: () => api.get('/api/v1/auth/profile'),
  updateProfile: (data) => api.put('/api/v1/auth/profile', data),
};

export const analysisAPI = {
  predict: (formData) => api.post('/api/v1/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  batchPredict: (formData) => api.post('/api/v1/predict/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  getHistory: (page = 1, limit = 10) => api.get(`/api/v1/analyses?page=${page}&limit=${limit}`),
  getAnalysis: (id) => api.get(`/api/v1/analyses/${id}`),
  deleteAnalysis: (id) => api.delete(`/api/v1/analyses/${id}`),
  exportAnalysis: (id) => api.get(`/api/v1/analyses/${id}/export`, { responseType: 'blob' }),
};

export const statsAPI = {
  getDashboard: () => api.get('/api/v1/stats/dashboard'),
  getAnalytics: (period = '7d') => api.get(`/api/v1/stats/analytics?period=${period}`),
};

export const userAPI = {
  getProfile: () => api.get('/api/v1/user/profile'),
  updateProfile: (data) => api.put('/api/v1/user/profile', data),
  deleteAccount: () => api.delete('/api/v1/user/account'),
};

export default api;