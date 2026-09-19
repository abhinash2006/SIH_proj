import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

export default function Viewer3D({ glbUrl, plyUrl, pointCloudUrl, metrics }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const currentModelRef = useRef(null);
  const trajectoryGroupRef = useRef(null);

  // Viewer State Toggles
  const [renderMode, setRenderMode] = useState('mesh'); // 'mesh' or 'points'
  const [showWireframe, setShowWireframe] = useState(false);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [measureMode, setMeasureMode] = useState(false);
  const [measurePoints, setMeasurePoints] = useState([]);
  const [measuredDistance, setMeasuredDistance] = useState(null);
  const [loadingModel, setLoadingModel] = useState(false);

  useEffect(() => {
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07090e');
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 2000);
    camera.position.set(20, 25, 35);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 1000;
    controls.minDistance = 1;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight1.position.set(50, 100, 50);
    dirLight1.castShadow = true;
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.8);
    dirLight2.position.set(-50, 50, -50);
    scene.add(dirLight2);

    // Ground Grid
    const grid = new THREE.GridHelper(100, 50, 0x00e5ff, 0x1e293b);
    grid.position.y = -0.1;
    scene.add(grid);

    // Trajectory Group
    const trajGroup = new THREE.Group();
    scene.add(trajGroup);
    trajectoryGroupRef.current = trajGroup;

    // Animation Loop
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (rendererRef.current && rendererRef.current.domElement) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  // Load Model when URL changes
  useEffect(() => {
    if (!sceneRef.current) return;

    // Clean up previous model
    if (currentModelRef.current) {
      sceneRef.current.remove(currentModelRef.current);
      currentModelRef.current = null;
    }

    const loader = new GLTFLoader();
    const modelToLoad = glbUrl;

    if (modelToLoad) {
      setLoadingModel(true);
      loader.load(
        modelToLoad,
        (gltf) => {
          const model = gltf.scene;
          model.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              if (child.material) {
                child.material.wireframe = showWireframe;
              }
            }
          });

          // Center model in view
          const box = new THREE.Box3().setFromObject(model);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          model.position.sub(center);

          sceneRef.current.add(model);
          currentModelRef.current = model;
          setLoadingModel(false);

          // Position camera
          const maxDim = Math.max(size.x, size.y, size.z, 5);
          cameraRef.current.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.8);
          controlsRef.current.target.set(0, 0, 0);
          controlsRef.current.update();

          // Build synthetic UAV flight trajectory above scene
          buildFlightTrajectory(maxDim);
        },
        undefined,
        (err) => {
          console.warn('Could not load GLB model, rendering procedural terrain model.', err);
          createProceduralFallback();
          setLoadingModel(false);
        }
      );
    } else {
      createProceduralFallback();
    }
  }, [glbUrl]);

  // Wireframe toggle effect
  useEffect(() => {
    if (currentModelRef.current) {
      currentModelRef.current.traverse((child) => {
        if (child.isMesh && child.material) {
          child.material.wireframe = showWireframe;
        }
      });
    }
  }, [showWireframe]);

  // Trajectory visibility effect
  useEffect(() => {
    if (trajectoryGroupRef.current) {
      trajectoryGroupRef.current.visible = showTrajectory;
    }
  }, [showTrajectory]);

  // Procedural Fallback Terrain Mesh for Demo
  const createProceduralFallback = () => {
    const scene = sceneRef.current;
    if (!scene) return;

    const group = new THREE.Group();
    const geo = new THREE.PlaneGeometry(30, 30, 40, 40);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const y = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 1.5 + Math.sin(x * 0.8) * 0.5;
      pos.setY(i, y);
    }
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.8,
      metalness: 0.2,
      wireframe: showWireframe,
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    scene.add(group);
    currentModelRef.current = group;
    buildFlightTrajectory(15);
  };

  // Build 3D flight trajectory curve and camera frustums
  const buildFlightTrajectory = (scale) => {
    const group = trajectoryGroupRef.current;
    if (!group) return;
    group.clear();

    const points = [];
    const numSteps = 24;
    for (let i = 0; i < numSteps; i++) {
      const t = i / (numSteps - 1);
      const angle = t * Math.PI * 1.5;
      const x = Math.cos(angle) * scale * 0.8;
      const z = Math.sin(angle) * scale * 0.8;
      const y = scale * 0.6 + Math.sin(t * Math.PI * 3) * (scale * 0.1);
      points.push(new THREE.Vector3(x, y, z));
    }

    // Flight line
    const curve = new THREE.CatmullRomCurve3(points);
    const lineGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(100));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 });
    const line = new THREE.Line(lineGeo, lineMat);
    group.add(line);

    // Camera cones at waypoints
    const coneGeo = new THREE.ConeGeometry(scale * 0.03, scale * 0.08, 4);
    coneGeo.rotateX(Math.PI);
    const coneMat = new THREE.MeshBasicMaterial({ color: 0x10b981, wireframe: true });

    points.forEach((pt, idx) => {
      if (idx % 3 === 0) {
        const cone = new THREE.Mesh(coneGeo, coneMat);
        cone.position.copy(pt);
        cone.lookAt(0, 0, 0);
        group.add(cone);
      }
    });
  };

  // Click handler for 3D Measurement Tool
  const handleCanvasClick = (e) => {
    if (!measureMode || !mountRef.current || !cameraRef.current || !sceneRef.current) return;

    const rect = mountRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current);

    const intersects = raycaster.intersectObjects(sceneRef.current.children, true);
    if (intersects.length > 0) {
      const pt = intersects[0].point;
      const newPts = [...measurePoints, pt];
      setMeasurePoints(newPts);

      if (newPts.length === 2) {
        const dist = newPts[0].distanceTo(newPts[1]);
        setMeasuredDistance(dist.toFixed(2));
      } else if (newPts.length > 2) {
        setMeasurePoints([pt]);
        setMeasuredDistance(null);
      }
    }
  };

  const resetView = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(20, 25, 35);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px', position: 'relative' }}>
      {/* Viewer Header & Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc' }}>Interactive 3D Digital Twin Viewer</h2>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Real-time Three.js WebGL rendering of reconstructed surface geometry</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setShowWireframe(!showWireframe)}
            className={`btn-secondary ${showWireframe ? 'active' : ''}`}
            style={{ borderColor: showWireframe ? '#00e5ff' : 'rgba(255, 255, 255, 0.1)' }}
          >
            {showWireframe ? '▦ Shaded' : '▤ Wireframe'}
          </button>

          <button
            onClick={() => setShowTrajectory(!showTrajectory)}
            className="btn-secondary"
            style={{ borderColor: showTrajectory ? '#10b981' : 'rgba(255, 255, 255, 0.1)' }}
          >
            {showTrajectory ? '🚁 Flight Path ON' : '🚁 Flight Path OFF'}
          </button>

          <button
            onClick={() => {
              setMeasureMode(!measureMode);
              setMeasurePoints([]);
              setMeasuredDistance(null);
            }}
            className="btn-secondary"
            style={{ borderColor: measureMode ? '#f59e0b' : 'rgba(255, 255, 255, 0.1)' }}
          >
            📏 {measureMode ? 'Measuring...' : 'Measure 3D'}
          </button>

          <button onClick={resetView} className="btn-secondary">
            ↺ Reset View
          </button>
        </div>
      </div>

      {/* Measurement Banner */}
      {measureMode && (
        <div style={{ padding: '8px 14px', background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem' }}>
          <span>Click 2 points on the 3D surface to measure Euclidean distance.</span>
          {measuredDistance && (
            <span style={{ fontWeight: 700, color: '#f59e0b', fontFamily: 'var(--font-mono)' }}>
              Distance: {measuredDistance} meters
            </span>
          )}
        </div>
      )}

      {/* 3D WebGL Canvas Mount */}
      <div
        ref={mountRef}
        onClick={handleCanvasClick}
        style={{
          width: '100%',
          height: '460px',
          borderRadius: '12px',
          overflow: 'hidden',
          position: 'relative',
          cursor: measureMode ? 'crosshair' : 'grab',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}
      >
        {loadingModel && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(10, 13, 20, 0.8)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            zIndex: 10,
          }}>
            <div className="pulsing-dot" style={{ width: '16px', height: '16px', backgroundColor: '#00e5ff' }} />
            <span style={{ fontSize: '0.85rem', color: '#00e5ff', fontWeight: 600 }}>Streaming 3D GLB Model...</span>
          </div>
        )}

        {/* Viewport Overlay Controls */}
        <div style={{ position: 'absolute', bottom: '12px', left: '12px', background: 'rgba(11, 15, 25, 0.8)', padding: '6px 12px', borderRadius: '6px', fontSize: '0.72rem', color: '#94a3b8', pointerEvents: 'none' }}>
          Orbit: Left Click + Drag • Pan: Right Click • Zoom: Scroll
        </div>

        {/* Export Download Links */}
        <div style={{ position: 'absolute', top: '12px', right: '12px', display: 'flex', gap: '6px' }}>
          {glbUrl && (
            <a href={glbUrl} download className="btn-secondary" style={{ padding: '6px 10px', fontSize: '0.75rem', textDecoration: 'none' }}>
              ⬇ GLB
            </a>
          )}
          {plyUrl && (
            <a href={plyUrl} download className="btn-secondary" style={{ padding: '6px 10px', fontSize: '0.75rem', textDecoration: 'none' }}>
              ⬇ PLY
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
