export default function StatsPanel({ stats }) {
  const { total, highRisk, byCls } = stats;
  const safeCount = byCls[0] || 0;
  const safeRate = total > 0 ? ((safeCount / total) * 100).toFixed(1) : '—';
  const riskRate = total > 0 ? ((highRisk / total) * 100).toFixed(1) : '—';

  const cards = [
    { label: 'Total Frames', value: total > 0 ? total.toLocaleString() : '0', color: '#e8e8f0' },
    { label: 'High-Risk Frames', value: highRisk > 0 ? highRisk.toLocaleString() : '0', color: '#ef4444' },
    { label: 'Safe Rate', value: total > 0 ? `${safeRate}%` : '—', color: '#22c55e' },
    { label: 'Risk Rate', value: total > 0 ? `${riskRate}%` : '—', color: '#f97316' },
  ];

  return (
    <div className="bg-card rounded-xl p-4 flex-shrink-0">
      <p className="text-[10px] font-semibold text-[#6b7280] uppercase tracking-wider mb-3">
        Session Stats
      </p>
      <div className="grid grid-cols-2 gap-2">
        {cards.map(({ label, value, color }) => (
          <div key={label} className="bg-panel rounded-lg px-3 py-2.5">
            <p className="text-[#6b7280] text-[10px] mb-0.5">{label}</p>
            <p className="text-lg font-bold leading-none tabular-nums" style={{ color }}>
              {value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
