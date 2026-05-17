import { SEV_COLOR } from '../constants';

export default function AlertFeed({ alerts }) {
  return (
    <div className="bg-card rounded-xl p-4 flex-shrink-0">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] font-semibold text-[#6b7280] uppercase tracking-wider">
          High-Risk Alerts
        </p>
        {alerts.length > 0 && (
          <span className="bg-danger/20 text-danger text-[10px] font-bold px-2 py-0.5 rounded-full">
            {alerts.length}
          </span>
        )}
      </div>

      {alerts.length === 0 ? (
        <p className="text-[#4a4a6a] text-xs text-center py-3">No alerts — keep driving safely!</p>
      ) : (
        <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
              style={{
                backgroundColor: SEV_COLOR[2] + '12',
                borderLeft: `3px solid ${SEV_COLOR[2]}`,
              }}
            >
              <span className="text-[#6b7280] font-mono flex-shrink-0 w-[52px]">
                {alert.ts}
              </span>
              <span className="text-[#e8e8f0] flex-1 truncate">{alert.class_name}</span>
              <span className="text-danger font-bold flex-shrink-0">
                {(alert.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
