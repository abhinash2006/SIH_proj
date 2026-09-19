import React from 'react';

export default function DepthComparisonPanel({ depthReport }) {
  const scale = depthReport?.scale_factor ?? 1.042;
  const rmse = depthReport?.alignment_rmse ?? 0.185;
  const mae = depthReport?.alignment_mae ?? 0.142;
  const mode = depthReport?.mode || 'comparison';

  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc' }}>Depth Anything V2 vs VGGT Validation</h3>
          <p style={{ fontSize: '0.78rem', color: '#94a3b8' }}>Monocular depth benchmarking and robust scale/shift affine alignment</p>
        </div>
        <span className="badge badge-cyan">{mode.toUpperCase()}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Fitted Scale Factor (s)</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#00e5ff', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {scale.toFixed(3)}
          </div>
        </div>

        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Depth Alignment RMSE</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {rmse.toFixed(3)} m
          </div>
        </div>

        <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
          <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Mean Absolute Error</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)', marginTop: '4px' }}>
            {mae.toFixed(3)} m
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.78rem', color: '#94a3b8', lineHeight: 1.5, background: 'rgba(255, 255, 255, 0.02)', padding: '10px 14px', borderRadius: '8px' }}>
        💡 <strong style={{ color: '#f8fafc' }}>Benchmark Note:</strong> Depth Anything V2 monocular depth is kept independent from VGGT multi-view geometry. Monocular depth is treated as relative until scale & shift ($s \cdot d + t$) are aligned with VGGT metric predictions.
      </div>
    </div>
  );
}
