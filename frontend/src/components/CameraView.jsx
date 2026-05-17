import { useEffect, useRef } from 'react';
import { SEV_COLOR, SEV_LABEL } from '../constants';

export default function CameraView({ videoRef, isCapturing, currentResult }) {
  const overlayRef = useRef(null);

  // Redraw detection overlay whenever result changes
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;

    const video = videoRef.current;
    const w = (canvas.width = video?.videoWidth || 640);
    const h = (canvas.height = video?.videoHeight || 480);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    if (!currentResult) return;

    const { class_name, confidence, severity, uncertain } = currentResult;
    const color = SEV_COLOR[severity] ?? '#22c55e';
    const label = uncertain ? 'Uncertain — reposition camera' : class_name;

    // ── top banner ──────────────────────────────────────────────────────────
    ctx.fillStyle = 'rgba(18,18,30,0.80)';
    ctx.fillRect(0, 0, w, 56);

    ctx.font = 'bold 16px Inter, system-ui, sans-serif';
    ctx.fillStyle = color;
    ctx.fillText(label, 14, 36);

    // ── bottom confidence bar ───────────────────────────────────────────────
    const bX = 12;
    const bY = h - 36;
    const bW = Math.min(w * 0.48, 220);
    const bH = 13;

    ctx.fillStyle = 'rgba(40,40,60,0.90)';
    ctx.beginPath();
    ctx.roundRect(bX, bY, bW, bH, 4);
    ctx.fill();

    if (confidence > 0) {
      ctx.fillStyle = color + 'cc';
      ctx.beginPath();
      ctx.roundRect(bX, bY, bW * confidence, bH, 4);
      ctx.fill();
    }

    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#e8e8f0';
    ctx.fillText(`${(confidence * 100).toFixed(1)}%`, bX + bW + 8, bY + 11);

    // ── severity pill (bottom-right) ────────────────────────────────────────
    const sevText = SEV_LABEL[severity] ?? 'SAFE';
    ctx.font = 'bold 10px Inter, system-ui, sans-serif';
    const tw = ctx.measureText(sevText).width;
    const pW = tw + 16;
    const pH = 22;
    const pX = w - pW - 10;
    const pY = h - 36;

    ctx.fillStyle = color + 'dd';
    ctx.beginPath();
    ctx.roundRect(pX, pY, pW, pH, 11);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillText(sevText, pX + 8, pY + 15);
  }, [currentResult, videoRef]);

  // Clear overlay when camera stops
  useEffect(() => {
    if (!isCapturing && overlayRef.current) {
      const ctx = overlayRef.current.getContext('2d');
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
  }, [isCapturing]);

  return (
    <div className="relative w-full h-full bg-[#0a0a14]">
      {/* Live video feed */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-cover"
      />

      {/* Detection overlay */}
      <canvas
        ref={overlayRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Placeholder when camera is off */}
      {!isCapturing && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0a0a14] gap-3">
          <div className="text-5xl select-none">📷</div>
          <p className="text-[#6b7280] text-sm font-medium">Camera is off</p>
          <p className="text-[#3a3a52] text-xs">
            Press <span className="text-accent font-semibold">Start Camera</span> to begin live detection
          </p>
        </div>
      )}
    </div>
  );
}
