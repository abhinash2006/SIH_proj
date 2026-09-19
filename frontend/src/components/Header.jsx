import React from 'react';

export default function Header({ systemInfo, activeMissionId, activeMode }) {
  return (
    <header className="glass-panel" style={{ margin: '16px 20px', padding: '14px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: '12px',
          background: 'linear-gradient(135deg, #00e5ff 0%, #3b82f6 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 15px rgba(0, 229, 255, 0.4)',
        }}>
          <span style={{ fontSize: '20px' }}>🚁</span>
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>AeroScan 3D</h1>
            <span className="badge badge-cyan">SIH UAV Reconstruction</span>
            <span className="badge badge-emerald">VGGT Foundation</span>
          </div>
          <p style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '2px' }}>
            Single-Pass Drone Video to Georeferenced 3D Digital Twin • YOLO + SAM 2 Dynamic Filtering
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        {activeMissionId && (
          <div style={{ textAlign: 'right', marginRight: '8px' }}>
            <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Active Mission</div>
            <div style={{ fontSize: '0.85rem', fontFamily: 'var(--font-mono)', color: '#00e5ff', fontWeight: 600 }}>{activeMissionId}</div>
          </div>
        )}

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '6px 14px',
          borderRadius: '10px',
        }}>
          <div className="pulsing-dot" style={{ backgroundColor: systemInfo?.cuda_available ? '#10b981' : '#f59e0b' }} />
          <div style={{ fontSize: '0.8rem', color: '#f8fafc', fontWeight: 500 }}>
            {systemInfo?.gpu_name || 'Detecting GPU...'}
          </div>
          {systemInfo?.cuda_available && (
            <span style={{ fontSize: '0.7rem', color: '#10b981', fontWeight: 600, background: 'rgba(16, 185, 129, 0.1)', padding: '2px 6px', borderRadius: '4px' }}>
              CUDA 12.4
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
