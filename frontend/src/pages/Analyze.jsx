import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  X,
  FileImage,
  Loader,
  Download,
  Share2,
  AlertCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { analysisAPI } from '../services/api';
import { useAnalysisStore } from '../store/useStore';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export default function Analyze() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const addAnalysis = useAnalysisStore((state) => state.addAnalysis);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map((file) =>
      Object.assign(file, {
        preview: URL.createObjectURL(file),
        id: Math.random().toString(36).substr(2, 9),
      })
    );
    setFiles((prev) => [...prev, ...newFiles]);
    setResults([]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    },
    maxSize: 10485760, // 10MB
  });

  const removeFile = (fileId) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
    setResults((prev) => prev.filter((r) => r.fileId !== fileId));
  };

  const handleAnalyze = async () => {
    if (files.length === 0) {
      toast.error('Please upload at least one image');
      return;
    }

    setLoading(true);
    const newResults = [];

    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
          const response = await analysisAPI.predict(formData);
          console.log('Prediction response:', response.data);
          newResults.push({
            fileId: file.id,
            fileName: file.name,
            preview: file.preview,
            ...response.data,
          });
          addAnalysis(response.data);
        } catch (error) {
          console.error('Prediction error:', error);
          console.error('Error response:', error.response);
          const errorMessage = error.response?.data?.detail 
            || error.message 
            || 'Analysis failed. Please check console for details.';
          newResults.push({
            fileId: file.id,
            fileName: file.name,
            preview: file.preview,
            error: errorMessage,
          });
          toast.error(`Failed to analyze ${file.name}: ${errorMessage}`);
        }
      }

      setResults(newResults);
      toast.success(`Analyzed ${newResults.length} image(s)`);
    } catch (error) {
      toast.error('Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const exportToPDF = async () => {
    // converts results to PDF - nst
    const element = document.getElementById('results-container');
    if (!element) return;

    try {
      const canvas = await html2canvas(element);
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('analysis-results.pdf');
      toast.success('Exported to PDF');
    } catch (error) {
      toast.error('Export failed');
    }
  };

  const reset = () => {
    files.forEach((file) => URL.revokeObjectURL(file.preview));
    setFiles([]);
    setResults([]);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Analyze Images</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Upload skin images for AI-powered diagnosis
        </p>
      </div>

          {/* drag and drop zone */}
      {files.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          {...getRootProps()}
          className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition ${
            isDragActive
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-primary-400 hover:bg-gray-50 dark:hover:bg-gray-800'
          }`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center">
              <Upload className="w-10 h-10 text-white" />
            </div>
            <div>
              <p className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                {isDragActive
                  ? 'Drop your images here'
                  : 'Drag & drop images or click to browse'}
              </p>
              <p className="text-gray-500 dark:text-gray-400">
                Supports: JPG, PNG, JPEG, WEBP (Max 10MB per file)
              </p>
            </div>
          </div>
        </motion.div>
      ) : (
        <>
          {/* show uploaded images in a grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
            {files.map((file, index) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className="relative group"
              >
                <div className="aspect-square rounded-lg overflow-hidden border-2 border-gray-200 dark:border-gray-700">
                  <img
                    src={file.preview}
                    alt={file.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <button
                  onClick={() => removeFile(file.id)}
                  className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition shadow-lg"
                >
                  <X className="w-4 h-4" />
                </button>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2 truncate">{file.name}</p>
              </motion.div>
            ))}

            {/* button to add more images -  */}
            <div
              {...getRootProps()}
              className="aspect-square rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 flex items-center justify-center cursor-pointer hover:border-primary-400 hover:bg-gray-50 dark:hover:bg-gray-800 transition"
            >
              <input {...getInputProps()} />
              <div className="text-center">
                <Upload className="w-8 h-8 text-gray-400 dark:text-gray-500 mx-auto mb-2" />
                <p className="text-sm text-gray-500 dark:text-gray-400">Add more</p>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="flex-1 bg-gradient-to-r from-primary-500 to-secondary-500 text-white py-4 rounded-xl font-semibold hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <FileImage className="w-5 h-5" />
                  Analyze {files.length} Image{files.length > 1 ? 's' : ''}
                </>
              )}
            </button>
            <button
              onClick={reset}
              className="px-6 py-4 border-2 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-xl font-semibold hover:bg-gray-50 dark:hover:bg-gray-800 transition"
            >
              Reset
            </button>
          </div>
        </>
      )}

      {/* Results Section */}
      {results.length > 0 && (
        <motion.div
          id="results-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Results</h2>
            <div className="flex gap-3">
              <button
                onClick={exportToPDF}
                className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition"
              >
                <Download className="w-4 h-4" />
                Export PDF
              </button>
              <button
                onClick={() => toast.success('Share feature coming soon!')}
                className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition"
              >
                <Share2 className="w-4 h-4" />
                Share
              </button>
            </div>
          </div>

          <div className="space-y-8">
            {results.map((result, index) => (
              <motion.div
                key={result.fileId}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
              >
                {result.error ? (
                  <div className="p-6">
                    <div className="flex items-start gap-3 text-red-600 dark:text-red-400">
                      <AlertCircle className="w-5 h-5 mt-0.5" />
                      <div>
                        <p className="font-semibold">Analysis Failed</p>
                        <p className="text-sm">{result.error}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
                    {/* Left Side - Images */}
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Original Image</h4>
                        <div className="aspect-square bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
                          <img
                            src={result.preview}
                            alt={result.fileName}
                            className="w-full h-full object-contain"
                          />
                        </div>
                      </div>

                      {/* Grad-CAM Visualization */}
                      {result.visualization && (
                        <div>
                          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                            </svg>
                            AI Attention Heatmap
                          </h4>
                          <div className="aspect-square bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
                            <img
                              src={`data:image/png;base64,${result.visualization.image}`}
                              alt="Grad-CAM Heatmap"
                              className="w-full h-full object-contain"
                            />
                          </div>
                          <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                            🔥 Red/yellow areas show regions the AI focused on. Generated from {result.visualization.num_models_visualized || 'multiple'} model(s).
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Right Side - Prediction Details */}
                    <div className="flex flex-col">
                      <div className="mb-6">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                            {result.predicted_class}
                          </h3>
                          <span className="px-4 py-2 bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 rounded-full text-base font-medium">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Primary Diagnosis</p>
                      </div>

                      {result.top_predictions && result.top_predictions.length > 1 && (
                        <div className="space-y-3 mb-6">
                          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                            Alternative Predictions:
                          </p>
                          {result.top_predictions.slice(1, 4).map((pred, idx) => (
                            <div
                              key={idx}
                              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                            >
                              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{pred.class}</span>
                              <span className="text-sm text-gray-600 dark:text-gray-400">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="mt-auto pt-6 border-t border-gray-200 dark:border-gray-700">
                        <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                          <p className="text-xs text-amber-900 dark:text-amber-200">
                            ⚕️ <strong>Medical Disclaimer:</strong> This is an AI-powered analysis for educational purposes.
                            Always consult a qualified healthcare professional for proper diagnosis and treatment.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
