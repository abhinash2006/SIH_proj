import React, { useState } from 'react';

export default function MissionUpload({ onUploadSuccess, isProcessing }) {
  const [videoFile, setVideoFile] = useState(null);
  const [gpsFile, setGpsFile] = useState(null);
  const [imuFile, setImuFile] = useState(null);
  const [cameraFile, setCameraFile] = useState(null);
  const [mode, setMode] = useState('hybrid');
  const [vggtBackend, setVggtBackend] = useState('pretrained');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile) {
      setUploadError('Please select a drone video file (MP4/MOV).');
      return;
    }

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('video', videoFile);
    if (gpsFile) formData.append('gps_csv', gpsFile);
    if (imuFile) formData.append('imu_csv', imuFile);
    if (cameraFile) formData.append('camera_json', cameraFile);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed with HTTP ${res.status}`);
      }

      const data = await res.json();
      onUploadSuccess({
        missionId: data.mission_id,
        mode: mode,
        vggtBackend: vggtBackend,
      });
    } catch (err) {
      setUploadError(err.message || 'Failed to upload flight mission files.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc' }}>Mission Data Ingestion</h2>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Provide UAV drone video, GPS telemetry, and flight parameters</p>
        </div>
        <span className="badge badge-cyan">Step 1: Staging</span>
      </div>

      {uploadError && (
        <div style={{ padding: '10px 16px', background: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '8px', color: '#fda4af', fontSize: '0.82rem' }}>
          ⚠️ {uploadError}
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Drone Video Input */}
        <div>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
            Drone Aerial Video (1080p / 4K MP4) <span style={{ color: '#00e5ff' }}>*</span>
          </label>
          <div style={{
            border: '2px dashed rgba(255, 255, 255, 0.12)',
            borderRadius: '10px',
            padding: '16px',
            textAlign: 'center',
            background: videoFile ? 'rgba(0, 229, 255, 0.04)' : 'rgba(255, 255, 255, 0.02)',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}>
            <input
              type="file"
              accept="video/*"
              id="video-input"
              style={{ display: 'none' }}
              onChange={(e) => setVideoFile(e.target.files[0] || null)}
            />
            <label htmlFor="video-input" style={{ cursor: 'pointer', display: 'block' }}>
              <div style={{ fontSize: '1.4rem', marginBottom: '4px' }}>🎬</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 500, color: videoFile ? '#00e5ff' : '#f8fafc' }}>
                {videoFile ? videoFile.name : 'Click to select or drop drone footage'}
              </div>
              <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '2px' }}>
                {videoFile ? `${(videoFile.size / (1024 * 1024)).toFixed(1)} MB` : 'MP4, MOV, AVI (H.264 / H.265)'}
              </div>
            </label>
          </div>
        </div>

        {/* Telemetry Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              GPS Log (CSV)
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setGpsFile(e.target.files[0] || null)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#94a3b8',
                fontSize: '0.8rem',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              IMU Sensors (CSV, Optional)
            </label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setImuFile(e.target.files[0] || null)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#94a3b8',
                fontSize: '0.8rem',
              }}
            />
          </div>
        </div>

        {/* Pipeline Configuration Selectors */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              Reconstruction Pipeline
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              style={{
                width: '100%',
                padding: '9px 12px',
                background: '#111622',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '0.82rem',
                outline: 'none',
              }}
            >
              <option value="hybrid">Mode 3: Hybrid (VGGT + YOLO/SAM2 + GPS) [Recommended]</option>
              <option value="vggt">Mode 1: Pure VGGT Geometry</option>
              <option value="colmap">Mode 2: Classical COLMAP Baseline</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#cbd5e1', marginBottom: '6px' }}>
              VGGT Model Checkpoint
            </label>
            <select
              value={vggtBackend}
              onChange={(e) => setVggtBackend(e.target.value)}
              style={{
                width: '100%',
                padding: '9px 12px',
                background: '#111622',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '0.82rem',
                outline: 'none',
              }}
            >
              <option value="pretrained">Pretrained Meta VGGT-1B (Official)</option>
              <option value="finetuned">Fine-Tuned UAV Checkpoint</option>
              <option value="mock">Synthetic Mock Adapter (Fast Demo)</option>
            </select>
          </div>
        </div>

        {/* Submit Action */}
        <button
          type="submit"
          disabled={!videoFile || uploading || isProcessing}
          className="btn-primary"
          style={{ width: '100%', justifyContent: 'center', marginTop: '6px', height: '44px' }}
        >
          {uploading ? (
            <span>Uploading UAV Flight Sequence...</span>
          ) : isProcessing ? (
            <span>Reconstruction In Progress...</span>
          ) : (
            <span>🚀 Initialize & Start 3D Reconstruction</span>
          )}
        </button>
      </form>
    </div>
  );
}
