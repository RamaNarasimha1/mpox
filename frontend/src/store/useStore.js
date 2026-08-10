import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => set({ user, token, isAuthenticated: true }),
      logout: () => set({ user: null, token: null, isAuthenticated: false }),
      updateUser: (user) => set({ user }),
    }),
    {
      name: 'dermavision-auth',
    }
  )
);

export const useThemeStore = create(
  persist(
    (set) => ({
      isDarkMode: false,
      toggleTheme: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
    }),
    {
      name: 'dermavision-theme',
    }
  )
);

export const useAnalysisStore = create((set) => ({
  analyses: [],
  currentAnalysis: null,
  addAnalysis: (analysis) =>
    set((state) => ({
      analyses: [analysis, ...state.analyses],
      currentAnalysis: analysis,
    })),
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  clearAnalyses: () => set({ analyses: [], currentAnalysis: null }),
}));
