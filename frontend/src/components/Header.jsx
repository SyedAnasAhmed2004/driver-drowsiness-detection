export default function Header({ isConnected, fps, total }) {
  return (
    <header className="flex-shrink-0 bg-panel border-b border-[#2e2e46] px-5 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="text-base font-bold text-[#e8e8f0]">
          Driver Distraction Detector
        </span>
        <span className="text-[10px] font-semibold bg-accent/20 text-accent px-2 py-0.5 rounded tracking-wide">
          MobileNetV2
        </span>
      </div>

      <div className="flex items-center gap-5 text-sm">
        {fps > 0 && (
          <span className="text-[#6b7280] tabular-nums">{fps} fps</span>
        )}
        {total > 0 && (
          <span className="text-[#6b7280] tabular-nums">
            {total.toLocaleString()} frames
          </span>
        )}

        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-safe shadow-[0_0_6px_#22c55e]' : 'bg-danger'
            }`}
          />
          <span
            className={`text-xs font-medium ${
              isConnected ? 'text-safe' : 'text-danger'
            }`}
          >
            {isConnected ? 'Backend connected' : 'Backend offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
