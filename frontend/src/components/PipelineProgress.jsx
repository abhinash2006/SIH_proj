import React, { useEffect, useRef } from 'react';

const STAGES = [
  { key: 'VIDEO', label: 'Video Frame Extraction' },
  { key: 'FILTER', label: 'Frame Quality Filter' },
  { key: 'KEYFRAME', label: 'Keyframe Selection' },
  { key: 'YOLO', label: 'YOLO Dynamic Detection' },
  { key: 'SAM2', label: 'SAM 2 Pixel Masking' },
  { key: 'VGGT', label: 'VGGT 3D Geometry' },
  { key: 'GEO', label: 'GPS / IMU Georeferencing' },
  { key: '3D', label: 'Open3D Point Cloud' },
  { key: 'MESH', label: 'Surface Reconstruction' },
  { key: 'EXPORT', label: 'GLB / OBJ / PLY Export' },
];

export default function PipelineProgress({ status, stage, progress, logs }) {
  const terminalRef = useRef(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const getCurrentStepIndex = () => {
    if (status === 'completed' || stage === 'COMPLETED') return STAGES.length;
    const idx = STAGES.findIndex((s) => s.key === stage);
    return idx >= 0 ? idx : 0;
  };

  const currentIdx = getCurrentStepIndex();

  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc' }}>Live Reconstruction Pipeline</h2>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Real-time telemetry and asynchronous execution logs</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className={`badge ${status === 'completed' ? 'badge-emerald' : status === 'failed' ? 'badge-rose' : 'badge-amber'}`}>
            {status.toUpperCase()}
          </span>
          <span style={{ fontSize: '0.85rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#00e5ff' }}>
            {progress}%
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
        <div
          style={{
            width: `${progress}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #00e5ff 0%, #3b82f6 50%, #10b981 100%)',
            transition: 'width 0.4s ease',
            boxShadow: '0 0 12px rgba(0, 229, 255, 0.5)',
          }}
        />
      </div>

      {/* Stage Flow Stepper */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px' }}>
        {STAGES.map((s, idx) => {
          const isDone = idx < currentIdx || status === 'completed';
          const isCurrent = idx === currentIdx && status === 'processing';

          return (
            <div
              key={s.key}
              style={{
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid',
                borderColor: isCurrent
                  ? '#00e5ff'
                  : isDone
                  ? 'rgba(16, 185, 129, 0.4)'
                  : 'rgba(255, 255, 255, 0.06)',
                background: isCurrent
                  ? 'rgba(0, 229, 255, 0.08)'
                  : isDone
                  ? 'rgba(16, 185, 129, 0.06)'
                  : 'rgba(255, 255, 255, 0.02)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s ease',
              }}
            >
              <div
                style={{
                  width: '20px',
                  height: '20px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  background: isDone ? '#10b981' : isCurrent ? '#00e5ff' : 'rgba(255, 255, 255, 0.1)',
                  color: '#0a0d14',
                }}
              >
                {isDone ? '✓' : idx + 1}
              </div>
              <div style={{ fontSize: '0.75rem', fontWeight: isCurrent ? 600 : 500, color: isCurrent ? '#00e5ff' : isDone ? '#f8fafc' : '#64748b' }}>
                {s.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* Live Terminal Log Viewer */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            System Console
          </span>
          <span style={{ fontSize: '0.7rem', color: '#64748b' }}>Streaming live logs</span>
        </div>
        <div
          ref={terminalRef}
          style={{
            background: '#070a10',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            borderRadius: '8px',
            padding: '12px 16px',
            height: '150px',
            overflowY: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            color: '#a7f3d0',
            lineHeight: 1.6,
          }}
        >
          {logs && logs.length > 0 ? (
            logs.map((log, i) => (
              <div key={i} style={{ color: log.includes('ERROR') ? '#f43f5e' : log.includes('WARNING') ? '#f59e0b' : '#38bdf8' }}>
                {log}
              </div>
            ))
          ) : (
            <div style={{ color: '#64748b' }}>Waiting for pipeline initialization...</div>
          )}
        </div>
      </div>
    </div>
  );
}
