import { useState, useRef, useEffect, useCallback } from 'react';
import Header from './components/Header';
import CameraView from './components/CameraView';
import UploadView from './components/UploadView';
import DetectionPanel from './components/DetectionPanel';
import AlertFeed from './components/AlertFeed';
import StatsPanel from './components/StatsPanel';
import ClassChart from './components/ClassChart';
import ThresholdControl from './components/ThresholdControl';
import { predictFrame, checkHealth, setThreshold as apiSetThreshold } from './api';
import { playDangerAlert, playMildAlert } from './utils/audio';

const EMPTY_STATS = { total: 0, highRisk: 0, byCls: {} };

export default function App() {
  const [activeTab, setActiveTab]         = useState('camera');
  const [isConnected, setIsConnected]     = useState(false);
  const [isCapturing, setIsCapturing]     = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [alerts, setAlerts]               = useState([]);
  const [stats, setStats]                 = useState(EMPTY_STATS);
  const [threshold, setThreshold]         = useState(0.6);
  const [fps, setFps]                     = useState(0);
  const [soundEnabled, setSoundEnabled]   = useState(true);

  const videoRef     = useRef(null);
  const streamRef    = useRef(null);
  const lastAlertRef = useRef(0);

  // ── backend health-check ───────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const data = await checkHealth();
        setIsConnected(data.status === 'ok');
      } catch {
        setIsConnected(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  // ── sync threshold to backend ──────────────────────────────────────────────
  useEffect(() => {
    apiSetThreshold(threshold).catch(() => {});
  }, [threshold]);

  // ── tab switch — stop camera + reset state ─────────────────────────────────
  const handleTabSwitch = useCallback((tab) => {
    if (tab === activeTab) return;
    // stop any active camera stream
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCapturing(false);
    setCurrentResult(null);
    setAlerts([]);
    setStats(EMPTY_STATS);
    setFps(0);
    setActiveTab(tab);
  }, [activeTab]);

  // ── camera start / stop ────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      setIsCapturing(true);
    } catch {
      alert('Camera access was denied.\nPlease allow camera access in your browser and try again.');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCapturing(false);
    setCurrentResult(null);
    setFps(0);
  }, []);

  // ── shared prediction handler (camera + video tabs both call this) ─────────
  const handlePrediction = useCallback((result) => {
    setCurrentResult(result);

    setStats((prev) => ({
      total:    prev.total + 1,
      highRisk: prev.highRisk + (result.severity === 2 ? 1 : 0),
      byCls:    { ...prev.byCls, [result.class]: (prev.byCls[result.class] || 0) + 1 },
    }));

    if (result.severity === 2 && result.confidence >= threshold && !result.uncertain) {
      setAlerts((prev) =>
        [{ ...result, id: Date.now(), ts: new Date().toLocaleTimeString() }, ...prev].slice(0, 20)
      );
    }

    // audio with cooldown
    if (soundEnabled && !result.uncertain) {
      const now      = Date.now();
      const cooldown = result.severity === 2 ? 3000 : 5000;
      if (now - lastAlertRef.current > cooldown) {
        if (result.severity === 2 && result.confidence >= threshold) {
          playDangerAlert();
          lastAlertRef.current = now;
        } else if (result.severity === 1 && result.confidence >= threshold) {
          playMildAlert();
          lastAlertRef.current = now;
        }
      }
    }
  }, [threshold, soundEnabled]);

  // ── live camera frame-capture loop ─────────────────────────────────────────
  useEffect(() => {
    if (!isCapturing) return;

    const captureCanvas = document.createElement('canvas');
    let inFlight  = false;
    let active    = true;
    let predCount = 0;
    let lastFpsTs = Date.now();

    const tick = setInterval(async () => {
      if (!active || inFlight) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || !video.videoWidth) return;

      captureCanvas.width  = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCanvas.getContext('2d').drawImage(video, 0, 0);

      captureCanvas.toBlob(async (blob) => {
        if (!blob || !active) return;
        inFlight = true;
        try {
          const result = await predictFrame(blob);
          if (!active) return;

          predCount++;
          const now = Date.now();
          if (now - lastFpsTs >= 1000) {
            setFps(Math.round(predCount / ((now - lastFpsTs) / 1000)));
            predCount = 0;
            lastFpsTs = now;
          }

          handlePrediction(result);
        } catch {
          // swallow network errors
        } finally {
          inFlight = false;
        }
      }, 'image/jpeg', 0.85);
    }, 150);

    return () => { active = false; clearInterval(tick); };
  }, [isCapturing, handlePrediction]);

  const resetStats = () => {
    setStats(EMPTY_STATS);
    setAlerts([]);
    setCurrentResult(null);
  };

  return (
    <div className="h-screen bg-dark text-[#e8e8f0] flex flex-col overflow-hidden font-sans">
      <Header isConnected={isConnected} fps={fps} total={stats.total} />

      <div className="flex flex-1 gap-3 p-3 overflow-hidden min-h-0">

        {/* ── left column ──────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-3 flex-[3] min-w-0">

          {/* Tab bar */}
          <div className="flex-shrink-0 flex gap-1 bg-card rounded-xl p-1">
            {[
              { id: 'camera', icon: '📷', label: 'Live Camera' },
              { id: 'video',  icon: '🎬', label: 'Upload' },
            ].map(({ id, icon, label }) => (
              <button
                key={id}
                onClick={() => handleTabSwitch(id)}
                className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-all ${
                  activeTab === id
                    ? 'bg-accent text-white shadow-[0_0_10px_rgba(124,92,191,0.35)]'
                    : 'text-[#6b7280] hover:text-[#e8e8f0] hover:bg-panel'
                }`}
              >
                <span>{icon}</span>
                <span>{label}</span>
              </button>
            ))}
          </div>

          {/* Main view */}
          <div className="flex-1 rounded-xl overflow-hidden min-h-0">
            {activeTab === 'camera' ? (
              <CameraView
                videoRef={videoRef}
                isCapturing={isCapturing}
                currentResult={currentResult}
              />
            ) : (
              <UploadView
                onPrediction={handlePrediction}
                currentResult={currentResult}
              />
            )}
          </div>

          {/* Controls bar */}
          <div className="flex-shrink-0 bg-card rounded-xl px-4 py-3 flex items-center gap-4">
            {activeTab === 'camera' && (
              <button
                onClick={isCapturing ? stopCamera : startCamera}
                className={`flex-shrink-0 px-5 py-2 rounded-lg text-sm font-semibold transition-all ${
                  isCapturing
                    ? 'bg-danger/15 text-danger hover:bg-danger/25 border border-danger/40'
                    : 'bg-accent hover:bg-accent/80 text-white shadow-[0_0_12px_rgba(124,92,191,0.4)]'
                }`}
              >
                {isCapturing ? '⏹ Stop Camera' : '▶ Start Camera'}
              </button>
            )}

            <ThresholdControl value={threshold} onChange={setThreshold} />

            <button
              onClick={() => setSoundEnabled((v) => !v)}
              className="flex-shrink-0 text-lg px-2 py-1 rounded-lg hover:bg-panel transition-colors"
              title={soundEnabled ? 'Mute alerts' : 'Unmute alerts'}
            >
              {soundEnabled ? '🔔' : '🔕'}
            </button>

            {stats.total > 0 && (
              <button
                onClick={resetStats}
                className="flex-shrink-0 text-xs text-[#6b7280] hover:text-[#e8e8f0] transition-colors"
                title="Reset session statistics"
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {/* ── right column: shared info panels ─────────────────────────────── */}
        <div className="flex flex-col gap-3 w-[320px] flex-shrink-0 overflow-y-auto">
          <DetectionPanel result={currentResult} threshold={threshold} />
          <AlertFeed alerts={alerts} />
          <StatsPanel stats={stats} />
          <ClassChart byCls={stats.byCls} />
        </div>
      </div>
    </div>
  );
}
