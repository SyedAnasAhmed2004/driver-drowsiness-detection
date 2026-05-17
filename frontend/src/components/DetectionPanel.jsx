import { SEV_COLOR, SEV_LABEL, CLASS_NAMES } from '../constants';

export default function DetectionPanel({ result, threshold }) {
  if (!result) {
    return (
      <div className="bg-card rounded-xl p-4 flex-shrink-0">
        <SectionLabel>Current Detection</SectionLabel>
        <p className="text-[#6b7280] text-sm text-center py-5">
          Waiting for camera…
        </p>
      </div>
    );
  }

  const { class_name, confidence, severity, uncertain, all_probs } = result;
  const color = SEV_COLOR[severity] ?? '#22c55e';
  const label = uncertain ? 'Uncertain — reposition camera' : class_name;

  // Top-4 classes by probability
  const topClasses = (all_probs ?? [])
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 4);

  return (
    <div className="bg-card rounded-xl p-4 flex-shrink-0">
      <SectionLabel>Current Detection</SectionLabel>

      {/* Severity badge */}
      <div
        className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold mb-2"
        style={{
          backgroundColor: color + '22',
          color,
          border: `1px solid ${color}55`,
        }}
      >
        {uncertain ? '?' : SEV_LABEL[severity]}
      </div>

      {/* Class name */}
      <p
        className="text-sm font-semibold leading-snug mb-3"
        style={{ color: uncertain ? '#6b7280' : color }}
      >
        {label}
      </p>

      {/* Confidence bar */}
      <div className="mb-1">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-[#6b7280]">Confidence</span>
          <span className="font-mono" style={{ color }}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-2 bg-panel rounded-full overflow-hidden relative">
          <div
            className="h-full rounded-full transition-all duration-200"
            style={{ width: `${confidence * 100}%`, backgroundColor: color }}
          />
          {/* Threshold marker */}
          <div
            className="absolute top-0 h-full w-px bg-[#6b7280]/60"
            style={{ left: `${threshold * 100}%` }}
            title={`Threshold: ${(threshold * 100).toFixed(0)}%`}
          />
        </div>
      </div>

      {/* Top-4 class breakdown */}
      <div className="mt-3 space-y-1.5">
        {topClasses.map(({ p, i }) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[#6b7280] text-[10px] font-mono w-5 text-right flex-shrink-0">
              c{i}
            </span>
            <div className="flex-1 h-1.5 bg-panel rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-200"
                style={{
                  width: `${p * 100}%`,
                  backgroundColor: i === result.class ? color : '#3a3a52',
                }}
              />
            </div>
            <span className="text-[#6b7280] text-[10px] font-mono w-9 text-right flex-shrink-0">
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <p className="text-[10px] font-semibold text-[#6b7280] uppercase tracking-wider mb-3">
      {children}
    </p>
  );
}
