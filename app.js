import { useState } from "react";
import "@/App.css";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import confetti from "canvas-confetti";
import CountUp from "react-countup";
import { Sparkles, Database, Zap, Brain, LineChart, TrendingUp, CheckCircle2, ArrowRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import {
  LineChart as RechartsLine,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass p-3 rounded-lg">
        <p className="text-sm font-mono text-cyan-400">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm text-slate-300">
            {entry.name}: {entry.value.toFixed(2)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

function App() {
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [dataInfo, setDataInfo] = useState(null);
  const [cleaningResults, setCleaningResults] = useState(null);
  const [edaResults, setEdaResults] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [predictionInput, setPredictionInput] = useState({
    MedInc: 3.5,
    HouseAge: 25,
    AveRooms: 5.5,
    AveBedrms: 1.2,
    Population: 1500,
    AveOccup: 3.0,
    Latitude: 37.5,
    Longitude: -122.3
  });
  const [predictionResult, setPredictionResult] = useState(null);

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#06b6d4', '#a855f7', '#f472b6']
    });
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data/info`);
      setDataInfo(response.data);
      setCurrentStep(1);
      toast.success("Dataset loaded successfully!");
      triggerConfetti();
    } catch (error) {
      toast.error("Failed to load data: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const cleanData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data/clean`);
      setCleaningResults(response.data);
      setCurrentStep(2);
      toast.success("Data cleaned successfully!");
    } catch (error) {
      toast.error("Failed to clean data: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const performEDA = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data/eda`);
      setEdaResults(response.data);
      setCurrentStep(3);
      toast.success("EDA completed!");
    } catch (error) {
      toast.error("Failed to perform EDA: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/train`);
      setModelResults(response.data);
      setCurrentStep(4);
      toast.success("Model trained successfully!");
      triggerConfetti();
    } catch (error) {
      toast.error("Failed to train model: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const makePrediction = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/model/predict`, predictionInput);
      setPredictionResult(response.data);
      toast.success("Prediction made successfully!");
      triggerConfetti();
    } catch (error) {
      toast.error("Failed to make prediction: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#030712] relative overflow-hidden">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-cyan-500/20 rounded-full blur-[100px] animate-float" />
        <div className="absolute top-40 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-[120px] animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-pink-500/20 rounded-full blur-[110px] animate-float" style={{ animationDelay: '4s' }} />
      </div>

      {/* Sticky Header */}
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="sticky top-0 z-50 glass-heavy border-b border-white/10"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center neon-glow">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-heading font-bold text-white">DataForge AI</h1>
                <p className="text-xs text-slate-400">Machine Learning Pipeline</p>
              </div>
            </div>
            <div className="flex gap-2">
              {[0, 1, 2, 3, 4].map((step) => (
                <div
                  key={step}
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    step <= currentStep ? "bg-cyan-500 w-8" : "bg-white/20"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </motion.header>

      {/* Hero Section */}
      <section className="relative py-20 md:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/30 mb-8">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span className="text-sm font-mono text-cyan-400">End-to-End ML Pipeline</span>
            </div>
            <h1 className="text-5xl md:text-7xl font-heading font-extrabold text-white mb-6 tracking-tight">
              Data Science
              <br />
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient">
                Made Beautiful
              </span>
            </h1>
            <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-12">
              Experience machine learning like never before. From raw data to predictions,
              visualized in stunning detail.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20 space-y-12">
        {/* Step 1: Load Dataset */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <Card className="rounded-3xl p-8 glass border-white/10 relative overflow-hidden group hover:border-cyan-500/30 transition-all duration-300" data-testid="step-1-card">
            <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-cyan-500/10 to-transparent rounded-full blur-3xl group-hover:scale-150 transition-transform duration-700" />
            <div className="relative z-10">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center neon-glow">
                  <Database className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h2 className="text-3xl font-heading font-bold text-white">Step 1: Load Dataset</h2>
                  <p className="text-slate-400">Import and explore the California Housing dataset</p>
                </div>
              </div>

              <Button
                onClick={loadData}
                disabled={loading}
                data-testid="load-data-btn"
                className="rounded-full bg-white text-black font-bold px-8 py-6 text-lg hover:scale-105 transition-transform duration-200 shadow-[0_0_20px_rgba(255,255,255,0.3)] mb-6"
              >
                {loading ? (
                  <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Loading...</>
                ) : (
                  <><Database className="w-5 h-5 mr-2" /> Load Dataset</>
                )}
              </Button>

              <AnimatePresence>
                {dataInfo && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="space-y-6"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="glass-heavy p-6 rounded-2xl border border-white/10"
                      >
                        <p className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-2">Total Samples</p>
                        <p className="text-4xl font-heading font-bold text-cyan-400">
                          <CountUp end={dataInfo.shape.rows} duration={2} separator="," />
                        </p>
                      </motion.div>
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="glass-heavy p-6 rounded-2xl border border-white/10"
                      >
                        <p className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-2">Features</p>
                        <p className="text-4xl font-heading font-bold text-purple-400">
                          <CountUp end={dataInfo.shape.columns} duration={2} />
                        </p>
                      </motion.div>
                      <motion.div
                        whileHover={{ scale: 1.02 }}
                        className="glass-heavy p-6 rounded-2xl border border-white/10"
                      >
                        <p className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-2">Missing Values</p>
                        <p className="text-4xl font-heading font-bold text-green-400">
                          <CountUp end={dataInfo.total_missing} duration={2} />
                        </p>
                      </motion.div>
                    </div>

                    <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                      <p className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-4">Features</p>
                      <div className="flex flex-wrap gap-2">
                        {dataInfo.columns.map((col, idx) => (
                          <motion.span
                            key={col}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: idx * 0.05 }}
                            className="px-4 py-2 rounded-full glass border border-cyan-500/30 text-sm font-mono text-cyan-400"
                          >
                            {col}
                          </motion.span>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </Card>
        </motion.div>

        {/* Step 2: Data Cleaning */}
        {currentStep >= 1 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <Card className="rounded-3xl p-8 glass border-white/10 relative overflow-hidden group hover:border-purple-500/30 transition-all duration-300" data-testid="step-2-card">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-purple-500/10 to-transparent rounded-full blur-3xl group-hover:scale-150 transition-transform duration-700" />
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-[0_0_20px_rgba(168,85,247,0.5)]">
                    <Zap className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-heading font-bold text-white">Step 2: Data Cleaning</h2>
                    <p className="text-slate-400">Handle missing values and detect outliers</p>
                  </div>
                </div>

                <Button
                  onClick={cleanData}
                  disabled={loading}
                  data-testid="clean-data-btn"
                  className="rounded-full bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold px-8 py-6 text-lg hover:scale-105 transition-transform duration-200 shadow-[0_0_20px_rgba(168,85,247,0.5)] mb-6"
                >
                  {loading ? (
                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Cleaning...</>
                  ) : (
                    <><Sparkles className="w-5 h-5 mr-2" /> Clean Data</>
                  )}
                </Button>

                <AnimatePresence>
                  {cleaningResults && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-4"
                    >
                      <div className="glass-heavy p-6 rounded-2xl border border-green-500/30">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="w-6 h-6 text-green-400" />
                          <p className="text-lg font-semibold text-green-400">{cleaningResults.summary}</p>
                        </div>
                      </div>

                      {cleaningResults.cleaning_steps.map((step, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="glass-heavy p-6 rounded-2xl border border-white/10 flex items-start gap-4"
                        >
                          <div className="w-10 h-10 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center flex-shrink-0">
                            <span className="text-lg font-mono font-bold text-purple-400">{idx + 1}</span>
                          </div>
                          <div className="flex-1">
                            <h3 className="text-lg font-semibold text-white mb-1">{step.step}</h3>
                            <p className="text-sm text-slate-400 mb-2">{step.method}</p>
                            <span className="inline-flex items-center px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-sm font-mono text-purple-400">
                              Count: {step.count.toLocaleString()}
                            </span>
                          </div>
                        </motion.div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Card>
          </motion.div>
        )}

        {/* Step 3: EDA */}
        {currentStep >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <Card className="rounded-3xl p-8 glass border-white/10 relative overflow-hidden group hover:border-pink-500/30 transition-all duration-300" data-testid="step-3-card">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-pink-500/10 to-transparent rounded-full blur-3xl group-hover:scale-150 transition-transform duration-700" />
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-pink-500 to-pink-600 flex items-center justify-center shadow-[0_0_20px_rgba(244,114,182,0.5)]">
                    <LineChart className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-heading font-bold text-white">Step 3: Exploratory Analysis</h2>
                    <p className="text-slate-400">Visualize patterns and correlations</p>
                  </div>
                </div>

                <Button
                  onClick={performEDA}
                  disabled={loading}
                  data-testid="perform-eda-btn"
                  className="rounded-full bg-gradient-to-r from-pink-500 to-orange-500 text-white font-bold px-8 py-6 text-lg hover:scale-105 transition-transform duration-200 shadow-[0_0_20px_rgba(244,114,182,0.5)] mb-6"
                >
                  {loading ? (
                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing...</>
                  ) : (
                    <><LineChart className="w-5 h-5 mr-2" /> Perform EDA</>
                  )}
                </Button>

                <AnimatePresence>
                  {edaResults && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-6"
                    >
                      <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                          <Sparkles className="w-5 h-5 text-pink-400" />
                          Key Insights
                        </h3>
                        <ul className="space-y-3">
                          {edaResults.key_insights.map((insight, idx) => (
                            <motion.li
                              key={idx}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: idx * 0.1 }}
                              className="flex items-start gap-3 text-slate-300"
                            >
                              <ArrowRight className="w-4 h-4 text-pink-400 mt-1 flex-shrink-0" />
                              <span>{insight}</span>
                            </motion.li>
                          ))}
                        </ul>
                      </div>

                      <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4">Correlation Heatmap</h3>
                        <img
                          src={edaResults.plots.correlation_heatmap}
                          alt="Correlation Heatmap"
                          className="w-full rounded-xl border border-white/10"
                        />
                      </div>

                      <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4">Feature Distributions</h3>
                        <img
                          src={edaResults.plots.distributions}
                          alt="Distributions"
                          className="w-full rounded-xl border border-white/10"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Card>
          </motion.div>
        )}

        {/* Step 4: Model Training */}
        {currentStep >= 3 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <Card className="rounded-3xl p-8 glass border-white/10 relative overflow-hidden group hover:border-cyan-500/30 transition-all duration-300" data-testid="step-4-card">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-cyan-500/10 to-transparent rounded-full blur-3xl group-hover:scale-150 transition-transform duration-700" />
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center neon-glow">
                    <Brain className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-heading font-bold text-white">Step 4: Model Training</h2>
                    <p className="text-slate-400">Train and evaluate ML models</p>
                  </div>
                </div>

                <Button
                  onClick={trainModel}
                  disabled={loading}
                  data-testid="train-model-btn"
                  className="rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold px-8 py-6 text-lg hover:scale-105 transition-transform duration-200 neon-glow mb-6"
                >
                  {loading ? (
                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Training...</>
                  ) : (
                    <><Brain className="w-5 h-5 mr-2" /> Train Models</>
                  )}
                </Button>

                <AnimatePresence>
                  {modelResults && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-6"
                    >
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <motion.div
                          whileHover={{ scale: 1.02 }}
                          className="glass-heavy p-8 rounded-2xl border border-white/10"
                        >
                          <h3 className="text-xl font-bold text-white mb-6">Linear Regression</h3>
                          <div className="space-y-4">
                            {Object.entries(modelResults.models.linear_regression).map(([key, value]) => (
                              <div key={key} className="flex justify-between items-center">
                                <span className="text-sm uppercase tracking-wider text-slate-400">{key}</span>
                                <span className="text-2xl font-mono font-bold text-cyan-400">
                                  {value.toFixed(4)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </motion.div>

                        <motion.div
                          whileHover={{ scale: 1.02 }}
                          className="glass-heavy p-8 rounded-2xl border border-purple-500/30 animate-pulse-glow"
                        >
                          <div className="flex items-center gap-2 mb-6">
                            <h3 className="text-xl font-bold text-white">Random Forest</h3>
                            <span className="px-2 py-1 rounded-full bg-green-500/20 border border-green-500/30 text-xs font-bold text-green-400">BEST</span>
                          </div>
                          <div className="space-y-4">
                            {Object.entries(modelResults.models.random_forest).map(([key, value]) => (
                              <div key={key} className="flex justify-between items-center">
                                <span className="text-sm uppercase tracking-wider text-slate-400">{key}</span>
                                <span className="text-2xl font-mono font-bold text-purple-400">
                                  {value.toFixed(4)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      </div>

                      <div className="glass-heavy p-6 rounded-2xl border border-green-500/30">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="w-6 h-6 text-green-400" />
                          <p className="text-lg font-semibold text-green-400">{modelResults.conclusion}</p>
                        </div>
                      </div>

                      <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4">Model Predictions vs Actual</h3>
                        <img
                          src={modelResults.plots.predictions}
                          alt="Predictions"
                          className="w-full rounded-xl border border-white/10"
                        />
                      </div>

                      <div className="glass-heavy p-6 rounded-2xl border border-white/10">
                        <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
                        <img
                          src={modelResults.plots.feature_importance}
                          alt="Feature Importance"
                          className="w-full rounded-xl border border-white/10"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Card>
          </motion.div>
        )}

        {/* Step 5: Predictions */}
        {currentStep >= 4 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <Card className="rounded-3xl p-8 glass border-white/10 relative overflow-hidden group hover:border-green-500/30 transition-all duration-300" data-testid="step-5-card">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-green-500/10 to-transparent rounded-full blur-3xl group-hover:scale-150 transition-transform duration-700" />
              <div className="relative z-10">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-[0_0_20px_rgba(34,197,94,0.5)]">
                    <TrendingUp className="w-7 h-7 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-heading font-bold text-white">Step 5: Make Predictions</h2>
                    <p className="text-slate-400">Use the trained model to predict house prices</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {Object.keys(predictionInput).map((key) => (
                    <div key={key}>
                      <Label className="text-xs font-mono uppercase tracking-widest text-slate-400 mb-2 block">
                        {key}
                      </Label>
                      <Input
                        id={key}
                        data-testid={`input-${key}`}
                        type="number"
                        step="0.1"
                        value={predictionInput[key]}
                        onChange={(e) => setPredictionInput({ ...predictionInput, [key]: parseFloat(e.target.value) })}
                        className="bg-white/5 border-white/10 text-white placeholder:text-slate-500 focus:border-cyan-500/50 focus:ring-cyan-500/20 rounded-xl font-mono"
                      />
                    </div>
                  ))}
                </div>

                <Button
                  onClick={makePrediction}
                  disabled={loading}
                  data-testid="predict-btn"
                  className="rounded-full bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold px-8 py-6 text-lg hover:scale-105 transition-transform duration-200 shadow-[0_0_20px_rgba(34,197,94,0.5)] mb-6 w-full md:w-auto"
                >
                  {loading ? (
                    <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Predicting...</>
                  ) : (
                    <><TrendingUp className="w-5 h-5 mr-2" /> Predict Price</>
                  )}
                </Button>

                <AnimatePresence>
                  {predictionResult && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="glass-heavy p-12 rounded-3xl border border-green-500/30 text-center relative overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-transparent" />
                      <div className="relative z-10">
                        <p className="text-sm font-mono uppercase tracking-widest text-slate-400 mb-4">Predicted House Value</p>
                        <motion.p
                          initial={{ scale: 0.5, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ delay: 0.2, type: "spring" }}
                          className="text-6xl md:text-8xl font-heading font-extrabold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-4"
                        >
                          {predictionResult.prediction_formatted}
                        </motion.p>
                        <p className="text-sm text-slate-400">Model: {predictionResult.model_used}</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Card>
          </motion.div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-white/10 glass-heavy mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center">
          <p className="text-slate-400">Built with DataForge AI • Machine Learning Made Beautiful</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
