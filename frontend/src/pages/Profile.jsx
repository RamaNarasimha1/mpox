import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  User,
  Mail,
  Calendar,
  Activity,
  Save,
  Camera,
  BarChart3,
  Clock,
  Award,
  Settings,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { userAPI, analysisAPI } from '../services/api';
import { useAnalysisStore, useAuthStore } from '../store/useStore';

export default function Profile() {
  const user = useAuthStore((state) => state.user);
  const updateUser = useAuthStore((state) => state.updateUser);
  const [profile, setProfile] = useState({
    name: user?.name || 'Guest User',
    email: user?.email || 'user@example.com',
    joinDate: user?.created_at || user?.createdAt || new Date().toISOString(),
    avatar: null,
  });
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    mostCommon: 'N/A',
    averageConfidence: 0,
    recentActivity: [],
  });

  const localAnalyses = useAnalysisStore((state) => state.analyses);

  useEffect(() => {
    loadProfile();
    loadStats();
  }, []);

  const loadProfile = async () => {
    setLoading(true);
    try {
      // First, set from auth store
      if (user) {
        setProfile({
          name: user.name || 'Guest User',
          email: user.email || 'user@example.com',
          joinDate: user.created_at || user.createdAt || new Date().toISOString(),
          avatar: null,
        });
      }
      
      // Try to fetch additional profile data from backend
      try {
        const response = await userAPI.getProfile();
        setProfile((prev) => ({ ...prev, ...response.data }));
      } catch (error) {
        // Fallback to localStorage if API fails
        const stored = localStorage.getItem('userProfile');
        if (stored) {
          const storedProfile = JSON.parse(stored);
          setProfile((prev) => ({ ...prev, ...storedProfile }));
        }
      }
    } catch (error) {
      console.error('Error loading profile:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await analysisAPI.getHistory(1, 1000);
      const analyses = response.data.items || [];
      calculateStats(analyses);
    } catch (error) {
      // Fallback to local storage
      calculateStats(localAnalyses);
    }
  };

  const calculateStats = (analyses) => {
    if (!analyses || analyses.length === 0) {
      setStats({
        totalAnalyses: 0,
        mostCommon: 'N/A',
        averageConfidence: 0,
        recentActivity: [],
      });
      return;
    }

    // Total analyses
    const total = analyses.length;

    // Most common prediction
    const classCounts = {};
    analyses.forEach((a) => {
      classCounts[a.predicted_class] = (classCounts[a.predicted_class] || 0) + 1;
    });
    const mostCommon =
      Object.keys(classCounts).length > 0
        ? Object.keys(classCounts).reduce((a, b) =>
            classCounts[a] > classCounts[b] ? a : b
          )
        : 'N/A';

    // Average confidence
    const avgConfidence =
      analyses.reduce((sum, a) => sum + (a.confidence || 0), 0) / total;

    // Recent activity (last 7 days)
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    const recent = analyses.filter(
      (a) => new Date(a.timestamp) >= sevenDaysAgo
    );

    setStats({
      totalAnalyses: total,
      mostCommon,
      averageConfidence: avgConfidence * 100,
      recentActivity: recent,
    });
  };

  const handleSave = async () => {
    try {
      // Update auth store
      updateUser({ ...user, name: profile.name, email: profile.email });
      
      // Try to update backend
      try {
        await userAPI.updateProfile(profile);
        toast.success('Profile updated successfully');
      } catch (error) {
        // Save locally as fallback
        localStorage.setItem('userProfile', JSON.stringify(profile));
        toast.success('Profile updated locally');
      }
      
      setIsEditing(false);
    } catch (error) {
      toast.error('Failed to update profile');
    }
  };

  const handleAvatarChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfile({ ...profile, avatar: reader.result });
      };
      reader.readAsDataURL(file);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Profile</h1>
        <p className="text-gray-600 mt-2">Manage your account and view statistics</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:col-span-1 bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <div className="flex flex-col items-center">
            {/* Avatar */}
            <div className="relative mb-4">
              <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center overflow-hidden">
                {profile.avatar ? (
                  <img
                    src={profile.avatar}
                    alt="Avatar"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <User className="w-16 h-16 text-white" />
                )}
              </div>
              {isEditing && (
                <label className="absolute bottom-0 right-0 bg-white rounded-full p-2 shadow-lg cursor-pointer hover:bg-gray-50 transition">
                  <Camera className="w-5 h-5 text-gray-700" />
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleAvatarChange}
                    className="hidden"
                  />
                </label>
              )}
            </div>

            {/* Name & Email */}
            {isEditing ? (
              <div className="w-full space-y-3 mb-4">
                <input
                  type="text"
                  value={profile.name}
                  onChange={(e) =>
                    setProfile({ ...profile, name: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="Name"
                />
                <input
                  type="email"
                  value={profile.email}
                  onChange={(e) =>
                    setProfile({ ...profile, email: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  placeholder="Email"
                />
              </div>
            ) : (
              <div className="text-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">
                  {profile.name}
                </h2>
                <p className="text-gray-600 mt-1">{profile.email}</p>
              </div>
            )}

            {/* Join Date */}
            <div className="flex items-center gap-2 text-sm text-gray-500 mb-6">
              <Calendar className="w-4 h-4" />
              <span>
                Joined {new Date(profile.joinDate).toLocaleDateString()}
              </span>
            </div>

            {/* Actions */}
            {isEditing ? (
              <div className="flex gap-2 w-full">
                <button
                  onClick={handleSave}
                  className="flex-1 bg-primary-500 text-white py-2 rounded-lg hover:bg-primary-600 transition flex items-center justify-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  Save
                </button>
                <button
                  onClick={() => {
                    setIsEditing(false);
                    loadProfile();
                  }}
                  className="flex-1 bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsEditing(true)}
                className="w-full bg-primary-500 text-white py-2 rounded-lg hover:bg-primary-600 transition flex items-center justify-center gap-2"
              >
                <Settings className="w-4 h-4" />
                Edit Profile
              </button>
            )}
          </div>
        </motion.div>

        {/* Stats Grid */}
        <div className="lg:col-span-2 space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center justify-between mb-2">
                <Activity className="w-8 h-8 text-primary-500" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {stats.totalAnalyses}
              </p>
              <p className="text-sm text-gray-600 mt-1">Total Analyses</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center justify-between mb-2">
                <Award className="w-8 h-8 text-green-500" />
              </div>
              <p className="text-3xl font-bold text-gray-900">
                {stats.averageConfidence.toFixed(1)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">Avg. Confidence</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center justify-between mb-2">
                <BarChart3 className="w-8 h-8 text-blue-500" />
              </div>
              <p className="text-2xl font-bold text-gray-900 truncate">
                {stats.mostCommon}
              </p>
              <p className="text-sm text-gray-600 mt-1">Most Common</p>
            </motion.div>
          </div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-gray-700" />
              <h3 className="text-lg font-semibold text-gray-900">
                Recent Activity (Last 7 Days)
              </h3>
            </div>

            {stats.recentActivity.length === 0 ? (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                <p className="text-gray-600">No recent activity</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {stats.recentActivity.slice(0, 10).map((activity, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition"
                  >
                    <div className="flex-1">
                      <p className="font-medium text-gray-900">
                        {activity.predicted_class}
                      </p>
                      <p className="text-sm text-gray-500">
                        {activity.timestamp
                          ? new Date(activity.timestamp).toLocaleString()
                          : 'Unknown date'}
                      </p>
                    </div>
                    <span className="px-3 py-1 bg-primary-100 text-primary-600 rounded-full text-sm font-medium">
                      {((activity.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          {/* Class Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-gray-700" />
              <h3 className="text-lg font-semibold text-gray-900">
                Prediction Distribution
              </h3>
            </div>

            {stats.totalAnalyses === 0 ? (
              <div className="text-center py-8">
                <BarChart3 className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                <p className="text-gray-600">No data to display</p>
              </div>
            ) : (
              <div className="space-y-3">
                {['Chickenpox', 'Measles', 'Monkeypox', 'Normal'].map(
                  (className) => {
                    const count = localAnalyses.filter(
                      (a) => a.predicted_class === className
                    ).length;
                    const percentage =
                      stats.totalAnalyses > 0
                        ? (count / stats.totalAnalyses) * 100
                        : 0;

                    return (
                      <div key={className}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium text-gray-700">
                            {className}
                          </span>
                          <span className="text-gray-600">
                            {count} ({percentage.toFixed(1)}%)
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${percentage}%` }}
                            transition={{ duration: 0.5, delay: 0.2 }}
                            className="bg-primary-500 h-2 rounded-full"
                          />
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
