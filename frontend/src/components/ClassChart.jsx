import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { CLASS_NAMES, SEVERITY, SEV_COLOR, SHORT_NAMES } from '../constants';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const { fullName, count } = payload[0].payload;
  return (
    <div
      className="text-xs rounded-lg px-3 py-2"
      style={{ background: '#1c1c2e', border: '1px solid #252540', color: '#e8e8f0' }}
    >
      <p className="font-semibold mb-0.5">{fullName}</p>
      <p className="text-[#6b7280]">{count} frames</p>
    </div>
  );
};

export default function ClassChart({ byCls }) {
  const data = CLASS_NAMES.map((name, i) => ({
    name: SHORT_NAMES[i],
    fullName: name,
    count: byCls[i] || 0,
    severity: SEVERITY[i],
  }));

  const total = Object.values(byCls).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-card rounded-xl p-4 flex-shrink-0">
      <p className="text-[10px] font-semibold text-[#6b7280] uppercase tracking-wider mb-3">
        Class Distribution
      </p>

      {total === 0 ? (
        <p className="text-[#4a4a6a] text-xs text-center py-4">No data yet</p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={data} margin={{ top: 2, right: 2, bottom: 0, left: -24 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#6b7280', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#6b7280', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(124,92,191,0.08)' }} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={SEV_COLOR[entry.severity] + 'bb'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div className="flex gap-4 mt-2 justify-center">
            {[['Safe', '#22c55e'], ['Mild Risk', '#f97316'], ['Danger', '#ef4444']].map(
              ([label, color]) => (
                <div key={label} className="flex items-center gap-1">
                  <div
                    className="w-2 h-2 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-[#6b7280] text-[10px]">{label}</span>
                </div>
              )
            )}
          </div>
        </>
      )}
    </div>
  );
}
