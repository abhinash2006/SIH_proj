import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import MissionUpload from './components/MissionUpload';
import PipelineProgress from './components/PipelineProgress';
import Viewer3D from './components/Viewer3D';
import DynamicObjectsPanel from './components/DynamicObjectsPanel';
import DepthComparisonPanel from './components/DepthComparisonPanel';
import MetricsCard from './components/MetricsCard';

export default function App() {
  const [systemInfo, setSystemInfo] = useState(null);
  const [activeMissionId, setActiveMissionId] = useState(null);
  const [activeMode, setActiveMode] = useState('hybrid');
  const [missionStatus, setMissionStatus] = useState('ready');
  const [stage, setStage] = useState('IDLE');
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([
    '[INIT] AeroScan 3D UAV Reconstruction Platform initialized.',
    '[READY] Waiting for drone flight sequence upload...',
  ]);
  const [artifacts, setArtifacts] = useState({
    glbUrl: null,
    plyUrl: null,
  });
  const [metrics, setMetrics] = useState(null);

  // Fetch backend health & GPU info on mount
  useEffect(() => {
    fetch('/api/')
      .then((res) => res.json())
      .then((data) => setSystemInfo(data))
      .catch((err) => console.log('Backend not yet connected:', err));
  }, []);

  // Poll mission progress while processing
  useEffect(() => {
    if (!activeMissionId || missionStatus !== 'processing') return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${activeMissionId}`);
        if (!res.ok) return;

        const data = await res.json();
        setMissionStatus(data.status);
        setStage(data.stage);
        setProgress(data.progress);
        if (data.logs && data.logs.length > 0) {
          setLogs(data.logs);
        }

        if (data.status === 'completed' || data.status === 'success' || data.status === 'completed_with_warnings') {
          clearInterval(interval);
          // Fetch artifacts
          const resArtifacts = await fetch(`/api/results/${activeMissionId}`);
          if (resArtifacts.ok) {
            const artData = await resArtifacts.json();
            setArtifacts({
              glbUrl: artData.artifacts.glb_url,
              plyUrl: artData.artifacts.ply_url,
            });
          }
          // Fetch metrics
          const resMetrics = await fetch(`/api/metrics/${activeMissionId}`);
          if (resMetrics.ok) {
            const mData = await resMetrics.json();
            setMetrics(mData);
          }
        }
      } catch (err) {
        console.warn('Polling error:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeMissionId, missionStatus]);

  // Handle successful staging & trigger background process
  const handleUploadSuccess = async ({ missionId, mode, vggtBackend }) => {
    setActiveMissionId(missionId);
    setActiveMode(mode);
    setMissionStatus('processing');
    setProgress(5);
    setStage('INITIALIZING');
    setLogs((prev) => [...prev, `[UPLOAD] Mission staged: '${missionId}'. Spawning reconstruction task...`]);

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mission_id: missionId,
          reconstruction_mode: mode,
          vggt_backend: vggtBackend,
        }),
      });

      if (!res.ok) {
        throw new Error('Failed to start processing job.');
      }
    } catch (err) {
      setMissionStatus('failed');
      setLogs((prev) => [...prev, `[ERROR] Failed to start pipeline: ${err.message}`]);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header
        systemInfo={systemInfo}
        activeMissionId={activeMissionId}
        activeMode={activeMode}
      />

      <main style={{ flex: 1, padding: '0 20px 30px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Top Control Grid: Upload & Progress */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: '20px' }}>
          <MissionUpload
            onUploadSuccess={handleUploadSuccess}
            isProcessing={missionStatus === 'processing'}
          />

          <PipelineProgress
            status={missionStatus}
            stage={stage}
            progress={progress}
            logs={logs}
          />
        </div>

        {/* 3D WebGL Viewer Panel */}
        <Viewer3D
          glbUrl={artifacts.glbUrl}
          plyUrl={artifacts.plyUrl}
          metrics={metrics}
        />

        {/* Dynamic Objects & Depth Comparison Panels */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <DynamicObjectsPanel />
          <DepthComparisonPanel />
        </div>

        {/* Quantitative Accuracy & Metrics Panel */}
        <MetricsCard metrics={metrics} />
      </main>
    </div>
  );
}
