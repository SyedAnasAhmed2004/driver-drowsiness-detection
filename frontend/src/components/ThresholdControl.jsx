export default function ThresholdControl({ value, onChange }) {
  return (
    <div className="flex items-center gap-3 flex-1 min-w-0">
      <span className="text-xs text-[#6b7280] font-medium whitespace-nowrap flex-shrink-0">
        Threshold
      </span>
      <input
        type="range"
        min={0.1}
        max={0.95}
        step={0.05}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 min-w-0"
      />
      <span className="text-xs text-[#e8e8f0] font-mono w-9 text-center flex-shrink-0 tabular-nums">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
