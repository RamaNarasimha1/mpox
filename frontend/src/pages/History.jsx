import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Calendar,
  Filter,
  Download,
  Trash2,
  Eye,
  Search,
  ChevronDown,
  FileText,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { analysisAPI } from '../services/api';
import { useAnalysisStore } from '../store/useStore';

export default function History() {
  const [analyses, setAnalyses] = useState([]);
  const [filteredAnalyses, setFilteredAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  const localAnalyses = useAnalysisStore((state) => state.analyses);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    filterAndSortAnalyses();
  }, [analyses, searchTerm, filterClass, sortBy]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      // Try to fetch from API
      const response = await analysisAPI.getHistory(1, 100);
      setAnalyses(response.data.items || []);
    } catch (error) {
      // Fallback to local storage
      console.log('Using local storage for history');
      setAnalyses(localAnalyses);
    } finally {
      setLoading(false);
    }
  };

  const filterAndSortAnalyses = () => {
    let filtered = [...analyses];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(
        (a) =>
          a.predicted_class?.toLowerCase().includes(searchTerm.toLowerCase()) ||
          a.analysis_id?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Class filter
    if (filterClass !== 'all') {
      filtered = filtered.filter((a) => a.predicted_class === filterClass);
    }

    // Sort
    if (sortBy === 'newest') {
      filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    } else if (sortBy === 'oldest') {
      filtered.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    } else if (sortBy === 'confidence') {
      filtered.sort((a, b) => b.confidence - a.confidence);
    }

    setFilteredAnalyses(filtered);
  };

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this analysis?')) return;

    try {
      await analysisAPI.deleteAnalysis(id);
      toast.success('Analysis deleted');
      loadHistory();
    } catch (error) {
      toast.error('Failed to delete analysis');
    }
  };

  const handleExport = async (id) => {
    try {
      const response = await analysisAPI.exportAnalysis(id);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `analysis_${id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Exported to PDF');
    } catch (error) {
      toast.error('Export failed');
    }
  };

  const classes = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal'];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Analysis History</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          View and manage all your past skin disease analyses
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search analyses..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          {/* Class Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              className="w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none"
            >
              <option value="all">All Classes</option>
              {classes.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
          </div>

          {/* Sort */}
          <div className="relative">
            <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent appearance-none"
            >
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="confidence">Highest Confidence</option>
            </select>
            <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
          </div>
        </div>
      </div>

      {/* Results Count */}
      <div className="mb-4 text-sm text-gray-600 dark:text-gray-400">
        Showing {filteredAnalyses.length} of {analyses.length} analyses
      </div>

      {/* Analysis List */}
      {filteredAnalyses.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
          <FileText className="w-16 h-16 mx-auto mb-4 text-gray-400 dark:text-gray-600" />
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            No analyses found
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {searchTerm || filterClass !== 'all'
              ? 'Try adjusting your filters'
              : 'Start by analyzing your first image!'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredAnalyses.map((analysis, index) => (
            <motion.div
              key={analysis.analysis_id || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden hover:shadow-md transition"
            >
              <div className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex gap-4 flex-1">
                    {/* Image thumbnail */}
                    <div className="w-24 h-24 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden">
                      {analysis.image_url ? (
                        <img
                          src={analysis.image_url}
                          alt={analysis.predicted_class}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <FileText className="w-8 h-8 text-gray-400 dark:text-gray-500" />
                      )}
                    </div>

                    {/* Details */}
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                            {analysis.predicted_class}
                          </h3>
                          <p className="text-sm text-gray-500 dark:text-gray-400">
                            {analysis.analysis_id || `Analysis #${index + 1}`}
                          </p>
                        </div>
                        <span className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 rounded-full text-sm font-medium">
                          {((analysis.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </div>

                      <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-3">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {analysis.timestamp
                            ? new Date(analysis.timestamp).toLocaleDateString()
                            : 'Unknown date'}
                        </span>
                        {analysis.image_name && (
                          <span className="truncate max-w-xs">
                            {analysis.image_name}
                          </span>
                        )}
                      </div>

                      {/* Top Predictions */}
                      {analysis.top_predictions && analysis.top_predictions.length > 1 && (
                        <div className="flex flex-wrap gap-2">
                          {analysis.top_predictions.slice(1, 3).map((pred, i) => (
                            <span
                              key={i}
                              className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded"
                            >
                              {pred.class_name || pred.class}: {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 ml-4">
                    <button
                      onClick={() => setSelectedAnalysis(analysis)}
                      className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
                      title="View Details"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleExport(analysis.analysis_id || index)}
                      className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
                      title="Export PDF"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                    <button
                      onClick={() => handleDelete(analysis.analysis_id || index)}
                      className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition"
                      title="Delete"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedAnalysis && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedAnalysis(null)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Analysis Details
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {selectedAnalysis.analysis_id}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedAnalysis(null)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Predicted Class
                  </label>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                    {selectedAnalysis.predicted_class}
                  </p>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Confidence
                  </label>
                  <p className="text-lg text-gray-900 dark:text-white mt-1">
                    {((selectedAnalysis.confidence || 0) * 100).toFixed(2)}%
                  </p>
                </div>

                {selectedAnalysis.top_predictions && (
                  <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                      All Predictions
                    </label>
                    <div className="space-y-2">
                      {selectedAnalysis.top_predictions.map((pred, i) => (
                        <div
                          key={i}
                          className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                        >
                          <span className="font-medium text-gray-900 dark:text-white">
                            {pred.class_name || pred.class}
                          </span>
                          <span className="text-gray-600 dark:text-gray-400">
                            {(pred.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Date</label>
                    <p className="text-gray-900 mt-1">
                      {selectedAnalysis.timestamp
                        ? new Date(selectedAnalysis.timestamp).toLocaleString()
                        : 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700">
                      Image Name
                    </label>
                    <p className="text-gray-900 mt-1 truncate">
                      {selectedAnalysis.image_name || 'N/A'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
