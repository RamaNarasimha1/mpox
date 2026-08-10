import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  Clock,
  FileText,
  Upload,
  BarChart3,
} from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { statsAPI, analysisAPI } from '../services/api';
import toast from 'react-hot-toast';

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe'];

export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    todayAnalyses: 0,
    averageConfidence: 0,
    topConditions: [],
  });
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [analytics, setAnalytics] = useState([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const [dashboardRes, historyRes, analyticsRes] = await Promise.all([
        statsAPI.getDashboard().catch(() => ({ data: null })),
        analysisAPI.getHistory(1, 5).catch(() => ({ data: { items: [] } })),
        statsAPI.getAnalytics('7d').catch(() => ({ data: [] })),
      ]);

      if (dashboardRes.data) {
        setStats(dashboardRes.data);
      }
      setRecentAnalyses(historyRes.data.items || []);
      setAnalytics(analyticsRes.data || []);
    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Analyses',
      value: stats.totalAnalyses,
      icon: FileText,
      color: 'bg-blue-500',
      change: stats.totalAnalyses > 0 ? 'All time' : 'No data',
    },
    {
      title: 'Today',
      value: stats.todayAnalyses,
      icon: Clock,
      color: 'bg-green-500',
      change: stats.todayAnalyses > 0 ? 'Today' : 'None yet',
    },
    {
      title: 'Avg Confidence',
      value: `${(stats.averageConfidence * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: 'bg-purple-500',
      change: stats.averageConfidence > 0.8 ? 'High' : stats.averageConfidence > 0.5 ? 'Medium' : 'Low',
    },
    {
      title: 'Active Models',
      value: '7',
      icon: Activity,
      color: 'bg-pink-500',
      change: 'Ensemble',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Welcome back! Here's your overview.</p>
        </div>
        <button
          onClick={() => navigate('/analyze')}
          className="flex items-center gap-2 bg-gradient-to-r from-primary-500 to-secondary-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition"
        >
          <Upload className="w-5 h-5" />
          New Analysis
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <span className="text-sm font-medium text-green-600 dark:text-green-400">{stat.change}</span>
            </div>
            <h3 className="text-gray-600 dark:text-gray-400 text-sm font-medium">{stat.title}</h3>
            <p className="text-3xl font-bold text-gray-900 dark:text-white mt-1">{stat.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Analytics Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Analysis Trend</h2>
            <BarChart3 className="w-5 h-5 text-gray-400" />
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={analytics.length > 0 ? analytics : [
              { date: 'Mon', count: 0 },
              { date: 'Tue', count: 0 },
              { date: 'Wed', count: 0 },
              { date: 'Thu', count: 0 },
              { date: 'Fri', count: 0 },
              { date: 'Sat', count: 0 },
              { date: 'Sun', count: 0 },
            ]}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#667eea" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Top Conditions */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700"
        >
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Top Conditions</h2>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={stats.topConditions.length > 0 ? stats.topConditions : [
                  { name: 'No data yet', value: 1 },
                ]}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {(stats.topConditions.length > 0 ? stats.topConditions : [
                  { name: 'No data yet', value: 1 },
                ]).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Recent Analyses */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700"
      >
        <div className="p-6 border-b border-gray-100 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Recent Analyses</h2>
            <button
              onClick={() => navigate('/history')}
              className="text-primary-500 hover:text-primary-600 dark:text-primary-400 dark:hover:text-primary-300 font-medium text-sm"
            >
              View All
            </button>
          </div>
        </div>
        <div className="divide-y divide-gray-100 dark:divide-gray-700">
          {recentAnalyses.length > 0 ? (
            recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700 transition cursor-pointer">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    {analysis.image_url && (
                      <img
                        src={analysis.image_url}
                        alt="Analysis"
                        className="w-16 h-16 rounded-lg object-cover"
                      />
                    )}
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">{analysis.predicted_class}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {new Date(analysis.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 rounded-full text-sm font-medium">
                      {(analysis.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="p-12 text-center text-gray-500 dark:text-gray-400">
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400 dark:text-gray-600" />
              <p>No analyses yet. Start by uploading your first image!</p>
              <button
                onClick={() => navigate('/analyze')}
                className="mt-4 text-primary-500 hover:text-primary-600 dark:text-primary-400 dark:hover:text-primary-300 font-medium"
              >
                Upload Image
              </button>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
