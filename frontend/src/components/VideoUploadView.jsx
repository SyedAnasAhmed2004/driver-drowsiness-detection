import { useState, useRef, useEffect } from 'react';
import { predictFrame } from '../api';
import { SEV_COLOR, SEV_LABEL } from '../constants';

export default function VideoUploadView({ onPrediction, currentResult }) {
  const videoRef     = useRef(null);
  const overlayRef   = useRef(null);
  const fileInputRef = useRef(null);

  const [videoUrl,    setVideoUrl]    = useState(null);
  const [fileName,    setFileName]    = useState('');
  const [isDetecting, setIsDetecting] = useState(false);

  // ── load file ──────────────────────────────────────────────────────────────
  const loadFile = (file) => {
    if (!file || !file.type.startsWith('video/')) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFileName(file.name);
    setVideoUrl(URL.createObjectURL(file));
    setIsDetecting(false);
  };

  const handleFileInput = (e) => loadFile(e.target.files?.[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    loadFile(e.dataTransfer.files?.[0]);
  };

  const removeVideo = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setFileName('');
    setIsDetecting(false);
    const canvas = overlayRef.current;
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  };

  // ── detection loop (while video is playing) ────────────────────────────────
  useEffect(() => {
    if (!isDetecting) return;

    const captureCanvas = document.createElement('canvas');
    let inFlight = false;
    let active   = true;

    const tick = setInterval(async () => {
      if (!active || inFlight) return;
      const video = videoRef.current;
      if (!video || video.paused || video.ended || !video.videoWidth) return;

      captureCanvas.width  = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCanvas.getContext('2d').drawImage(video, 0, 0);

      captureCanvas.toBlob(async (blob) => {
        if (!blob || !active) return;
        inFlight = true;
        try {
          const result = await predictFrame(blob);
          if (active) onPrediction(result);
        } catch {
          // network / server error
        } finally {
          inFlight = false;
        }
      }, 'image/jpeg', 0.85);
    }, 200);

    return () => {
      active = false;
      clearInterval(tick);
    };
  }, [isDetecting, onPrediction]);

  // ── draw overlay on canvas ─────────────────────────────────────────────────
  useEffect(() => {
    const canvas = overlayRef.current;
    const video  = videoRef.current;
    if (!canvas || !currentResult) return;

    const w = (canvas.width  = video?.videoWidth  || 640);
    const h = (canvas.height = video?.videoHeight || 480);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    const { class_name, confidence, severity, uncertain } = currentResult;
    const color = SEV_COLOR[severity] ?? '#22c55e';
    const label = uncertain ? 'Uncertain — reposition view' : class_name;

    // top banner
    ctx.fillStyle = 'rgba(18,18,30,0.82)';
    ctx.fillRect(0, 0, w, 54);

    ctx.font      = 'bold 15px Inter, system-ui, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(label, 14, 34);

    ctx.font      = '13px Inter, system-ui, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(`${(confidence * 100).toFixed(1)}%`, w - 60, 34);

    // severity pill (top-right)
    const sevText = SEV_LABEL[severity] ?? 'SAFE';
    ctx.font = 'bold 10px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(sevText).width;
    const pW = tw + 14, pH = 20, pX = w - pW - 10, pY = 7;
    ctx.fillStyle = color + 'dd';
    ctx.beginPath();
    ctx.roundRect(pX, pY, pW, pH, 10);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(sevText, pX + 7, pY + 14);
  }, [currentResult]);

  // ── render ─────────────────────────────────────────────────────────────────
  if (!videoUrl) {
    return (
      <div
        className="w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-[#2e2e46] rounded-xl cursor-pointer hover:border-accent/50 hover:bg-accent/5 transition-all"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <span className="text-5xl mb-4 select-none">🎬</span>
        <p className="text-[#e8e8f0] text-sm font-semibold mb-1">
          Drop a video file here
        </p>
        <p className="text-[#6b7280] text-xs">or click to browse</p>
        <p className="text-[#3a3a52] text-xs mt-3">MP4 · AVI · MOV · MKV</p>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col bg-black rounded-xl overflow-hidden">
      {/* video + overlay */}
      <div className="flex-1 relative min-h-0">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="w-full h-full object-contain"
          onPlay={() => setIsDetecting(true)}
          onPause={() => setIsDetecting(false)}
          onEnded={() => setIsDetecting(false)}
        />
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* file info bar */}
      <div className="flex-shrink-0 flex items-center justify-between bg-card px-4 py-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-[#6b7280]">🎬</span>
          <span className="text-xs text-[#e8e8f0] truncate">{fileName}</span>
          {isDetecting && (
            <span className="text-[10px] bg-safe/20 text-safe px-2 py-0.5 rounded-full font-semibold flex-shrink-0">
              DETECTING
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-xs text-accent hover:text-accent/80 transition-colors"
          >
            Change
          </button>
          <button
            onClick={removeVideo}
            className="text-xs text-[#6b7280] hover:text-danger transition-colors"
          >
            Remove
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    </div>
  );
}
