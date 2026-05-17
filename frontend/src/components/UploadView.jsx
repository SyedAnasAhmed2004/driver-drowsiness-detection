import { useState, useRef, useEffect, useCallback } from 'react';
import { predictFrame } from '../api';
import { SEV_COLOR, SEV_LABEL } from '../constants';

function drawOverlay(ctx, w, h, result) {
  const { class_name, confidence, severity, uncertain } = result;
  const color = SEV_COLOR[severity] ?? '#22c55e';
  const label = uncertain ? 'Uncertain — reposition view' : class_name;

  // top banner
  ctx.fillStyle = 'rgba(18,18,30,0.82)';
  ctx.fillRect(0, 0, w, 54);

  ctx.font = 'bold 15px Inter, system-ui, sans-serif';
  ctx.fillStyle = color;
  ctx.fillText(label, 14, 34);

  ctx.font = '13px Inter, system-ui, sans-serif';
  ctx.fillText(`${(confidence * 100).toFixed(1)}%`, w - 62, 34);

  // severity pill
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
}

export default function UploadView({ onPrediction, currentResult }) {
  const fileInputRef  = useRef(null);
  const videoRef      = useRef(null);
  const videoOverlay  = useRef(null); // canvas over the video
  const imageCanvas   = useRef(null); // canvas that renders image + overlay
  const loadedImg     = useRef(null); // HTMLImageElement cache

  const [fileUrl,     setFileUrl]     = useState(null);
  const [fileName,    setFileName]    = useState('');
  const [fileType,    setFileType]    = useState(null); // 'image' | 'video'
  const [isDetecting, setIsDetecting] = useState(false);

  // ── load a new file ────────────────────────────────────────────────────────
  const loadFile = useCallback(async (file) => {
    if (!file) return;
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    if (!isImage && !isVideo) return;

    if (fileUrl) URL.revokeObjectURL(fileUrl);
    const url = URL.createObjectURL(file);
    setFileName(file.name);
    setFileUrl(url);
    setFileType(isImage ? 'image' : 'video');
    setIsDetecting(false);
    loadedImg.current = null;

    if (isImage) {
      // load HTMLImageElement then run one prediction
      const img = new Image();
      img.onload = async () => {
        loadedImg.current = img;
        try {
          const result = await predictFrame(file);
          onPrediction(result);
        } catch { /* ignore */ }
      };
      img.src = url;
    }
  }, [fileUrl, onPrediction]);

  const handleFileInput = (e) => loadFile(e.target.files?.[0]);
  const handleDrop = (e) => { e.preventDefault(); loadFile(e.dataTransfer.files?.[0]); };

  const removeFile = () => {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFileUrl(null); setFileName(''); setFileType(null);
    setIsDetecting(false); loadedImg.current = null;
  };

  // ── video detection loop ───────────────────────────────────────────────────
  useEffect(() => {
    if (!isDetecting || fileType !== 'video') return;
    const cap = document.createElement('canvas');
    let inFlight = false, active = true;

    const tick = setInterval(async () => {
      if (!active || inFlight) return;
      const v = videoRef.current;
      if (!v || v.paused || v.ended || !v.videoWidth) return;
      cap.width = v.videoWidth; cap.height = v.videoHeight;
      cap.getContext('2d').drawImage(v, 0, 0);
      cap.toBlob(async (blob) => {
        if (!blob || !active) return;
        inFlight = true;
        try { const r = await predictFrame(blob); if (active) onPrediction(r); }
        catch { /* ignore */ } finally { inFlight = false; }
      }, 'image/jpeg', 0.85);
    }, 200);

    return () => { active = false; clearInterval(tick); };
  }, [isDetecting, fileType, onPrediction]);

  // ── draw video overlay ─────────────────────────────────────────────────────
  useEffect(() => {
    if (fileType !== 'video' || !currentResult) return;
    const canvas = videoOverlay.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;
    const w = canvas.width  = video.videoWidth  || 640;
    const h = canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    drawOverlay(ctx, w, h, currentResult);
  }, [currentResult, fileType]);

  // ── draw image canvas (image + overlay combined) ───────────────────────────
  useEffect(() => {
    if (fileType !== 'image') return;
    const canvas = imageCanvas.current;
    const img    = loadedImg.current;
    if (!canvas || !img) return;

    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    if (currentResult) drawOverlay(ctx, canvas.width, canvas.height, currentResult);
  }, [currentResult, fileType]);

  // ── upload zone ────────────────────────────────────────────────────────────
  if (!fileUrl) {
    return (
      <div
        className="w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-[#2e2e46] rounded-xl cursor-pointer hover:border-accent/50 hover:bg-accent/5 transition-all select-none"
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="flex gap-3 text-4xl mb-4">🖼️ 🎬</div>
        <p className="text-[#e8e8f0] text-sm font-semibold mb-1">
          Drop a file here
        </p>
        <p className="text-[#6b7280] text-xs">or click to browse</p>
        <div className="flex gap-4 mt-4">
          <span className="flex items-center gap-1.5 text-xs text-[#3a3a52] bg-panel px-3 py-1.5 rounded-lg">
            🖼️ JPG · JPEG · PNG · BMP · WEBP
          </span>
          <span className="flex items-center gap-1.5 text-xs text-[#3a3a52] bg-panel px-3 py-1.5 rounded-lg">
            🎬 MP4 · AVI · MOV · MKV
          </span>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>
    );
  }

  // ── image viewer ───────────────────────────────────────────────────────────
  if (fileType === 'image') {
    return (
      <div className="w-full h-full flex flex-col bg-black rounded-xl overflow-hidden">
        <div className="flex-1 flex items-center justify-center bg-[#0a0a14] min-h-0 p-2">
          <canvas
            ref={imageCanvas}
            className="max-w-full max-h-full rounded"
          />
        </div>
        <FileBar
          icon="🖼️" fileName={fileName}
          badge={currentResult ? 'ANALYZED' : 'ANALYZING…'}
          badgeClass={currentResult ? 'bg-accent/20 text-accent' : 'bg-[#6b7280]/20 text-[#6b7280]'}
          onChangeClick={() => fileInputRef.current?.click()}
          onRemove={removeFile}
          fileInputRef={fileInputRef}
          onFileInput={handleFileInput}
        />
      </div>
    );
  }

  // ── video viewer ───────────────────────────────────────────────────────────
  return (
    <div className="w-full h-full flex flex-col bg-black rounded-xl overflow-hidden">
      <div className="flex-1 relative min-h-0">
        <video
          ref={videoRef}
          src={fileUrl}
          controls
          className="w-full h-full object-contain"
          onPlay={() => setIsDetecting(true)}
          onPause={() => setIsDetecting(false)}
          onEnded={() => setIsDetecting(false)}
        />
        <canvas
          ref={videoOverlay}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
      </div>
      <FileBar
        icon="🎬" fileName={fileName}
        badge={isDetecting ? 'DETECTING' : null}
        badgeClass="bg-safe/20 text-safe"
        onChangeClick={() => fileInputRef.current?.click()}
        onRemove={removeFile}
        fileInputRef={fileInputRef}
        onFileInput={handleFileInput}
      />
    </div>
  );
}

// ── small reusable bottom bar ─────────────────────────────────────────────────
function FileBar({ icon, fileName, badge, badgeClass, onChangeClick, onRemove, fileInputRef, onFileInput }) {
  return (
    <div className="flex-shrink-0 flex items-center justify-between bg-card px-4 py-2">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-xs text-[#6b7280]">{icon}</span>
        <span className="text-xs text-[#e8e8f0] truncate">{fileName}</span>
        {badge && (
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold flex-shrink-0 ${badgeClass}`}>
            {badge}
          </span>
        )}
      </div>
      <div className="flex items-center gap-3 flex-shrink-0">
        <button onClick={onChangeClick} className="text-xs text-accent hover:text-accent/80 transition-colors">
          Change
        </button>
        <button onClick={onRemove} className="text-xs text-[#6b7280] hover:text-danger transition-colors">
          Remove
        </button>
      </div>
      <input ref={fileInputRef} type="file" accept="image/*,video/*" onChange={onFileInput} className="hidden" />
    </div>
  );
}
