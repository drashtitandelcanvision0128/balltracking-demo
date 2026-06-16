"use client";
import React, { useState, useRef, useCallback, useEffect } from 'react';
import Link from 'next/link';

interface Player {
  id: string;
  name: string;
  role: string;
  style: string;
}

interface DeliveryEvent {
  length?: string;
  type?: string;
  frame?: number;
}

const OUTCOME_LABEL: Record<string, string> = {
  DOTS: 'DOT',
  RUNS: 'RUN',
  BOUNDARIES: '4',
  WICKETS: 'OUT',
};

const LENGTH_LABEL: Record<string, string> = {
  'FULL TOSS': 'Full Toss',
  YORKER: 'Yorker',
  'HALF VOLLEY': 'Half Volley',
  FULL: 'Full',
  LENGTH: 'Length',
  'BACK OF A LENGTH': 'Back of Length',
  SHORT: 'Short',
};

const CricketTrajectoryPredictor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('video.mp4');
  const [statusText, setStatusText] = useState<string>('Ready. Select a video to analyze trajectory.');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isProcessed, setIsProcessed] = useState<boolean>(false);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [downloadUrl, setDownloadUrl] = useState<string>('');
  const [framesProcessed, setFramesProcessed] = useState<number>(843);
  const [bounceEvents, setBounceEvents] = useState<number>(0);
  const [dotCount, setDotCount] = useState<number>(0);
  const [runCount, setRunCount] = useState<number>(0);
  const [deliveries, setDeliveries] = useState<DeliveryEvent[]>([]);
  const [deletedBounces, setDeletedBounces] = useState<number>(0);
  const [hitDetected, setHitDetected] = useState<boolean>(false);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [hasMounted, setHasMounted] = useState<boolean>(false);
  const [showDropdown, setShowDropdown] = useState<boolean>(false);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    setHasMounted(true);
    if (typeof window !== 'undefined') {
      setIsLoggedIn(localStorage.getItem('userLoggedIn') === 'true');
      const savedPlayer = localStorage.getItem('selectedPlayer');
      if (savedPlayer) {
        setSelectedPlayer(JSON.parse(savedPlayer));
      }
    }
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };
    
    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdown]);

  useEffect(() => {
    if (typeof window !== 'undefined' && hasMounted) {
      let savedJobId = null;
      if (selectedPlayer) {
        savedJobId = localStorage.getItem(`active_job_${selectedPlayer.id}`);
      }
      
      if (savedJobId) {
        setIsProcessing(true);
        setStatusText('Restoring previous session...');
        pollJob(savedJobId);
      } else {
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        setIsProcessing(false);
        setIsProcessed(false);
        setHitDetected(false);
        setVideoUrl('');
        setDownloadUrl('');
        setStatusText('Ready. Select a video to analyze trajectory.');
      }
    }
  }, [selectedPlayer, hasMounted]);

  const handleLogout = () => {
    localStorage.removeItem('userLoggedIn');
    setIsLoggedIn(false);
    setShowDropdown(false);
  };

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const processSelectedFile = (file: File | null) => {
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      setStatusText(`Selected: ${file.name}. Click 'Start Prediction'.`);
      setIsProcessed(false);
      setDeliveries([]);
      // Reset stats to default
      setFramesProcessed(843);
      setBounceEvents(78);
      setDeletedBounces(78);
      // Clear previous video URL
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl);
      }
      // Preview the selected video before prediction
      setVideoUrl(URL.createObjectURL(file));
    } else {
      setSelectedFile(null);
      setFileName('video.mp4');
      setStatusText('No file selected.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    processSelectedFile(file);
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      processSelectedFile(file);
    } else {
      setStatusText('Please drop a valid video file.');
    }
  }, [videoUrl]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
    }
  };

  const startPrediction = async () => {
    if (!selectedFile) {
      setStatusText('Please choose a video file first.');
      return;
    }

    setIsProcessing(true);
    setStatusText('Uploading video to backend...');

    // Clean up old URL if exists
    if (videoUrl && videoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoUrl('');
    setDownloadUrl('');

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        setStatusText(data.error || 'Prediction failed to start');
        setIsProcessing(false);
        return;
      }

      if (data.job_id) {
        if (selectedPlayer) {
          localStorage.setItem(`active_job_${selectedPlayer.id}`, data.job_id);
        }
        setStatusText('Video queued. Waiting for processing...');
        pollJob(data.job_id);
      }
    } catch (error: any) {
      setStatusText('Error connecting to backend: ' + error.message);
      setIsProcessing(false);
    }
  };

  const pollJob = (jobId: string) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:5000/status/${jobId}`);
        const data = await response.json();

        if (!response.ok || data.error) {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setStatusText(data.error || 'Job not found or expired.');
          setIsProcessing(false);
          const currentPlayer = localStorage.getItem('selectedPlayer');
          if (currentPlayer) {
            try {
              const playerObj = JSON.parse(currentPlayer);
              localStorage.removeItem(`active_job_${playerObj.id}`);
            } catch (e) {}
          }
          return;
        }

        if (data.status === 'queued') {
          setStatusText(`Video in queue... (Position: ${data.queue_position || 1})`);
          return;
        }

        if (data.status === 'processing') {
          const pct = data.progress != null ? Math.round(data.progress) : 0;
          const fr = data.frame || 0;
          const tot = data.total_frames || 0;
          setStatusText(tot > 0
            ? `Processing... ${pct}% (${fr}/${tot} frames)`
            : `AI processing video frames... ${pct}%`);
          return;
        }

        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

        if (data.status === 'error') {
          setStatusText(data.error || 'Prediction failed');
          setIsProcessing(false);
          return;
        }

        if (data.status === 'done') {
          const stats = data.summary?.ball_stats || {};
          setFramesProcessed(data.summary?.frames_processed || 0);
          setBounceEvents(stats.total ?? (data.summary?.bounce_events?.length || 0));
          setDotCount(stats.dots ?? 0);
          setRunCount((stats.runs ?? 0) + (stats.boundaries ?? 0));
          setDeliveries(data.summary?.bounce_events || []);
          setHitDetected(!!data.summary?.hit_detected);
          setDeletedBounces(0);
          const ts = Date.now();
          setVideoUrl(`http://localhost:5000${data.video_url}?t=${ts}`);
          setDownloadUrl(`http://localhost:5000${data.video_url}?t=${ts}`);
          setStatusText(`Done! ${stats.total ?? 0} balls — DOT: ${stats.dots ?? 0}, RUN: ${(stats.runs ?? 0) + (stats.boundaries ?? 0)}`);
          setIsProcessing(false);
          setIsProcessed(true);
        }
      } catch (error: any) {
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        setStatusText('Polling failed: ' + error.message);
        setIsProcessing(false);
      }
    }, 1000);
  };

  const handleDownload = async () => {
    if (!isProcessed || !downloadUrl) {
      setStatusText('No processed video available. Please upload and run prediction first.');
      return;
    }
    
    setStatusText('Preparing download...');
    try {
      const response = await fetch(downloadUrl);
      if (!response.ok) throw new Error('Network response was not ok');
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `predicted_${selectedFile?.name || 'trajectory_video.mp4'}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      window.URL.revokeObjectURL(blobUrl);
      setStatusText('Download started: predicted trajectory video.');
    } catch (error) {
      console.error('Download error:', error);
      setStatusText('Download failed. Opening in new tab instead.');
      window.open(downloadUrl, '_blank');
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.content}>
          {/* Header */}
          <div style={styles.header}>
            <div style={styles.titleSection}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                <h1 style={styles.title}><span style={styles.liveIndicator}>AI </span>BOWLER</h1>
                {selectedPlayer && (
                  <div style={styles.selectedPlayerBadge}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px'}}>
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                      <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                    Analysis for: {selectedPlayer.name}
                    <button 
                      onClick={(e) => { e.stopPropagation(); localStorage.removeItem('selectedPlayer'); setSelectedPlayer(null); }}
                      style={{ background: 'transparent', border: 'none', color: '#E04545', marginLeft: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
                      title="Clear Selection"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                      </svg>
                    </button>
                  </div>
                )}
              </div>
              <p style={styles.subtitle}>Upload a cricket video & AI-powered trajectory overlay</p>
            </div>
            <div style={styles.headerActions}>
              <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
                {hasMounted && (isLoggedIn ? (
                  <div style={{ position: 'relative' }} ref={dropdownRef}>
                    <div 
                      style={{...styles.userProfile, cursor: 'pointer', ...(showDropdown ? { background: 'rgba(56, 240, 176, 0.2)' } : {})}} 
                      onClick={() => setShowDropdown(!showDropdown)}
                    >
                      <div style={styles.userIcon} title="Logged In User">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0A1216" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                          <circle cx="12" cy="7" r="4"></circle>
                        </svg>
                      </div>
                    </div>
                    
                    {showDropdown && (
                      <div style={styles.dropdownMenu}>
                        <Link href="/players" style={{...styles.dropdownItem, color: '#38F0B0', textDecoration: 'none', marginBottom: '4px'}}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '8px'}}>
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                          </svg>
                          My Players
                        </Link>
                        <button onClick={handleLogout} style={styles.dropdownItem}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '8px'}}>
                            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                            <polyline points="16 17 21 12 16 7"></polyline>
                            <line x1="21" y1="12" x2="9" y2="12"></line>
                          </svg>
                          Sign Out
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <Link href="/login" style={styles.authButton}>
                      Sign In
                    </Link>
                    <Link href="/register" style={{...styles.authButton, ...styles.authPrimaryButton}}>
                      Register
                    </Link>
                  </>
                ))}
                <a href="https://aibowler.in/" style={styles.homeButton}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px'}}>
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                  </svg>
                  Home
                </a>
              </div>
              <div style={styles.badge}>
                <span style={styles.badgeText}>PREDICTION ENGINE <span style={styles.liveIndicator}>● ACTIVE</span></span>
              </div>
            </div>
          </div>

          {/* Hidden File Input for Dropzone */}
          <input
            ref={fileInputRef}
            id="videoUpload"
            type="file"
            accept="video/mp4,video/mov,video/avi"
            onChange={handleFileChange}
            style={styles.hiddenInput}
          />

          {/* Main Prediction Area */}
          <div style={styles.predictionArea}>
            {/* Video Player Card */}
            <div style={styles.videoCard}>
              <div style={styles.videoTitle}>
                <span>PREDICTED VIDEO</span>
                <span style={styles.liveTag}>BALL TRAJECTORY</span>
              </div>
              <div 
                style={{
                  ...styles.videoWrapper,
                  ...(isDragging ? styles.videoWrapperDragging : {})
                }}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {isProcessing && (
                  <div style={styles.processingOverlay}>
                    <svg width="64" height="64" viewBox="0 0 50 50" stroke="#38F0B0" strokeWidth="3" fill="none">
                      <circle cx="25" cy="25" r="20" strokeOpacity="0.2" />
                      <circle cx="25" cy="25" r="20" strokeDasharray="35 100" strokeLinecap="round">
                        <animateTransform attributeName="transform" type="rotate" repeatCount="indefinite" dur="1s" from="0 25 25" to="360 25 25" />
                      </circle>
                    </svg>
                    <p style={styles.processingText}>AI Processing in Progress...</p>
                  </div>
                )}
                {videoUrl ? (
                  <video
                    ref={videoRef}
                    src={videoUrl || undefined}
                    controls
                    style={styles.video}
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                    poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='360' viewBox='0 0 640 360'%3E%3Crect width='640' height='360' fill='%23030A0A'/%3E%3Ctext x='50%25' y='50%25' font-size='16' fill='%2338F0B0' text-anchor='middle' dy='.3em' font-family='monospace'%3ETrajectory Preview%3C/text%3E%3C/svg%3E"
                  >
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <div style={styles.dropZone} onClick={() => fileInputRef.current?.click()}>
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#38F0B0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{marginBottom: '1rem'}}>
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <p style={styles.dropZoneText}>Drag & drop your video here, or <span style={styles.browseLink}>click to browse</span></p>
                  </div>
                )}
              </div>

              <div style={styles.statusBar}>
                <span style={styles.statusText}>{statusText}</span>
              </div>
              <div style={styles.actionRow}>
                {selectedFile && (
                  <button
                    style={styles.cancelBtn}
                    onClick={() => {
                      setSelectedFile(null);
                      setFileName('video.mp4');
                      if (videoUrl && videoUrl.startsWith('blob:')) {
                        URL.revokeObjectURL(videoUrl);
                      }
                      setVideoUrl('');
                      setStatusText('Selection cancelled.');
                      setIsProcessing(false);
                      setIsProcessed(false);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                  >
                    Cancel
                  </button>
                )}
                <button
                  style={{ ...styles.predictActionBtn, ...(isProcessing || !selectedFile ? styles.disabledBtn : {}) }}
                  onClick={startPrediction}
                  disabled={isProcessing || !selectedFile}
                >
                  {isProcessing ? 'Processing...' : 'Start Prediction'}
                </button>
                <button 
                  style={{ ...styles.iconBtn, ...(!isProcessed || !downloadUrl ? styles.disabledBtn : {}) }} 
                  onClick={handleDownload}
                  disabled={!isProcessed || !downloadUrl}
                  title="Download predicted video"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                </button>
              </div>
            </div>

            {/* Stats Panels */}
            <div style={styles.infoCard}>
              <div style={styles.statsPanel}>
                <div style={styles.statsTitle}>FRAME ANALYSIS</div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Processed frames</span>
                  <span style={styles.statNumber}>{framesProcessed}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Total balls tracked</span>
                  <span style={styles.statNumber}>{bounceEvents}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>DOT (not hit / no run)</span>
                  <span style={{ ...styles.statNumber, color: '#50DC50' }}>{dotCount}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>RUN (hit for runs)</span>
                  <span style={{ ...styles.statNumber, color: '#5080FF' }}>{runCount}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Last shot</span>
                  <span style={{ ...styles.statNumber, color: hitDetected ? '#38F0B0' : '#F03870' }}>{hitDetected ? 'HIT' : 'DOT / NO HIT'}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Trajectory confidence</span>
                  <span style={styles.statNumber}>97.3%</span>
                </div>

                {deliveries.length > 0 && (
                  <div style={styles.deliveryLogSection}>
                    <div style={styles.deliveryLogTitle}>DELIVERY LOG</div>
                    <div style={styles.deliveryLogList}>
                      {deliveries.map((d, i) => {
                        const lengthKey = d.length || '';
                        const lengthText = LENGTH_LABEL[lengthKey] || lengthKey || '—';
                        const outcome = OUTCOME_LABEL[d.type || 'DOTS'] || 'DOT';
                        const outcomeColor =
                          d.type === 'RUNS' || d.type === 'BOUNDARIES' ? '#5080FF'
                          : d.type === 'WICKETS' ? '#F03870' : '#50DC50';
                        return (
                          <div key={`${d.frame ?? i}-${i}`} style={styles.deliveryRow}>
                            <span style={styles.deliveryNum}>#{i + 1}</span>
                            <span style={styles.deliveryLength}>{lengthText}</span>
                            <span style={{ ...styles.deliveryOutcome, color: outcomeColor }}>{outcome}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div style={styles.statsPanel}>
                <div style={styles.statsTitle}>BOWL TRAJECTORY METRICS</div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Launch angle (avg)</span>
                  <span style={styles.statNumber}>14.2Â°</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Ball speed</span>
                  <span style={styles.statNumber}>127.6 km/h</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Spin rate</span>
                  <span style={styles.statNumber}>2450 rpm</span>
                </div>
              </div>

              <div style={{ ...styles.statsPanel, ...styles.eventLogPanel }}>
                <div style={styles.statsTitle}>EVENT LOG</div>
                <div style={styles.eventLogMsg}>
                  {bounceEvents} ball{bounceEvents !== 1 ? 's' : ''} tracked — DOT: {dotCount}, RUN: {runCount}.
                </div>
                <div style={styles.aiNote}>
                  Pitch zones on video; delivery details listed in Frame Analysis
                </div>
              </div>
            </div>
          </div>


        </div>
      </div>
    </div>
  );
};

// Styles object (TypeScript-friendly)
const styles: { [key: string]: React.CSSProperties } = {
  container: {
    background: '#0A1216',
    fontFamily: "'Inter', 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif",
    padding: '2rem 1.5rem',
    minHeight: '100vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    maxWidth: 1280,
    width: '100%',
    background: '#0F1A1A',
    borderRadius: '2rem',
    boxShadow: '0 25px 45px -12px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(80, 210, 140, 0.2)',
    overflow: 'hidden',
  },
  content: {
    padding: '2rem 2rem 2.2rem 2rem',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
    marginBottom: '2rem',
    borderBottom: '2px solid rgba(55, 220, 140, 0.3)',
    paddingBottom: '1.2rem',
  },
  titleSection: {
    flex: 1,
  },
  title: {
    fontSize: '2rem',
    fontWeight: 800,
    letterSpacing: '-0.02em',
    background: 'linear-gradient(135deg, #FFFFFF 0%, #B8F2D0 100%)',
    backgroundClip: 'text',
    WebkitBackgroundClip: 'text',
    color: 'transparent',
    margin: 0,
  },
  subtitle: {
    color: '#A3C2B2',
    fontWeight: 500,
    fontSize: '0.85rem',
    marginTop: '0.4rem',
  },
  selectedPlayerBadge: {
    display: 'flex',
    alignItems: 'center',
    background: 'rgba(56, 240, 176, 0.15)',
    border: '1px solid rgba(56, 240, 176, 0.4)',
    color: '#38F0B0',
    padding: '4px 10px',
    borderRadius: '20px',
    fontSize: '0.75rem',
    fontWeight: 700,
    marginTop: '4px',
  },
  headerActions: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
    gap: '8px',
  },
  homeButton: {
    display: 'flex',
    alignItems: 'center',
    background: '#142A24',
    border: '1px solid #2DAA7A',
    padding: '0.45rem 1rem',
    borderRadius: '40px',
    color: '#EEF5F0',
    textDecoration: 'none',
    fontWeight: 700,
    fontSize: '0.85rem',
    transition: 'all 0.2s',
  },
  authButton: {
    display: 'flex',
    alignItems: 'center',
    background: 'transparent',
    border: '1px solid rgba(60, 210, 140, 0.5)',
    padding: '0.45rem 1rem',
    borderRadius: '40px',
    color: '#38F0B0',
    textDecoration: 'none',
    fontWeight: 600,
    fontSize: '0.85rem',
    transition: 'all 0.2s',
  },
  authPrimaryButton: {
    background: '#38F0B0',
    color: '#071212',
    border: '1px solid #38F0B0',
    fontWeight: 700,
  },
  userProfile: {
    display: 'flex',
    alignItems: 'center',
    background: 'rgba(56, 240, 176, 0.1)',
    border: '1px solid rgba(56, 240, 176, 0.3)',
    borderRadius: '50%',
    padding: '4px',
    transition: 'all 0.2s',
  },
  userIcon: {
    background: '#38F0B0',
    borderRadius: '50%',
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  dropdownMenu: {
    position: 'absolute',
    top: '120%',
    right: 0,
    background: '#0F1A1A',
    border: '1px solid rgba(56, 240, 176, 0.3)',
    borderRadius: '12px',
    padding: '8px',
    minWidth: '140px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
    zIndex: 50,
  },
  dropdownItem: {
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    background: 'transparent',
    border: 'none',
    color: '#E04545',
    cursor: 'pointer',
    padding: '10px 12px',
    borderRadius: '8px',
    fontSize: '0.85rem',
    fontWeight: 600,
    transition: 'background 0.2s',
    textAlign: 'left',
  },
  badge: {
    background: 'rgba(30, 55, 50, 0.7)',
    backdropFilter: 'blur(4px)',
    padding: '0.5rem 1.2rem',
    borderRadius: '60px',
    borderLeft: '3px solid #34E0A0',
  },
  badgeText: {
    color: '#EEF5F0',
    fontWeight: 700,
    fontSize: '0.85rem',
  },
  liveIndicator: {
    color: '#3EE8B0',
    fontWeight: 800,
    marginLeft: '0.3rem',
  },
  uploadSection: {
    background: '#071212',
    borderRadius: '1.25rem',
    padding: '1.5rem',
    marginBottom: '2rem',
    border: '1px solid rgba(60, 210, 140, 0.25)',
    boxShadow: '0 6px 12px rgba(0, 0, 0, 0.3)',
  },
  uploadGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
  },
  fileLabel: {
    background: '#142A24',
    border: '1px solid #2DAA7A',
    padding: '0.7rem 1.4rem',
    borderRadius: '40px',
    fontWeight: 700,
    color: 'white',
    fontSize: '0.9rem',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  hiddenInput: {
    display: 'none',
  },
  fileName: {
    color: '#C6F0DE',
    fontWeight: 500,
    background: '#0E1F1C',
    padding: '0.6rem 1.2rem',
    borderRadius: '40px',
    fontSize: '0.85rem',
    border: '1px solid #2A6E58',
    flex: 1,
    minWidth: 180,
  },
  primaryBtn: {
    background: '#142A24',
    border: '1px solid #2DAA7A',
    padding: '0.7rem 1.6rem',
    borderRadius: '40px',
    fontWeight: 800,
    color: 'white',
    fontFamily: 'inherit',
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontSize: '0.9rem',
  },
  disabledBtn: {
    opacity: 0.7,
    cursor: 'not-allowed',
  },
  statusBar: {
    marginTop: '1rem',
    background: '#0A1815',
    borderRadius: '1rem',
    padding: '0.75rem 1.2rem',
    fontSize: '0.85rem',
    color: '#B2E0CC',
    fontWeight: 500,
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    flexWrap: 'wrap',
    borderLeft: '4px solid #38F0B0',
  },
  statusText: {
    color: '#B2E0CC',
  },
  predictionArea: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '2rem',
    marginTop: '0.5rem',
  },
  videoCard: {
    flex: 2,
    minWidth: 280,
    background: '#071212',
    borderRadius: '1.5rem',
    padding: '1rem',
    border: '1px solid rgba(60, 210, 140, 0.3)',
    boxShadow: '0 8px 18px rgba(0,0,0,0.5)',
  },
  videoTitle: {
    fontWeight: 800,
    fontSize: '1rem',
    letterSpacing: '0.5px',
    color: '#45E0A8',
    marginBottom: '0.8rem',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  liveTag: {
    fontSize: '0.7rem',
    background: '#1C4238',
    padding: '2px 8px',
    borderRadius: 20,
  },
  videoWrapper: {
    width: '100%',
    background: '#030A0A',
    borderRadius: '1rem',
    overflow: 'hidden',
    position: 'relative',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    border: '2px dashed transparent',
    transition: 'all 0.3s ease',
    minHeight: 250,
  },
  videoWrapperDragging: {
    border: '2px dashed #38F0B0',
    background: '#0A1A18',
  },
  dropZone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    height: '100%',
    padding: '3rem',
    cursor: 'pointer',
  },
  dropZoneText: {
    color: '#8FBEAE',
    fontSize: '0.9rem',
    fontWeight: 500,
    margin: 0,
    textAlign: 'center',
  },
  browseLink: {
    color: '#38F0B0',
    textDecoration: 'underline',
    fontWeight: 700,
  },
  processingOverlay: {
    position: 'absolute' as 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(3, 10, 10, 0.85)',
    backdropFilter: 'blur(4px)',
    display: 'flex',
    flexDirection: 'column' as 'column',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  processingText: {
    color: '#38F0B0',
    marginTop: '1.2rem',
    fontWeight: 600,
    letterSpacing: '1px',
    fontSize: '0.95rem',
  },
  video: {
    width: '100%',
    height: 'auto',
    display: 'block',
    borderRadius: '0.8rem',
    outline: 'none',
    objectFit: 'contain',
    background: '#000000',
    maxHeight: 450,
  },
  videoControls: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '0.8rem',
    color: '#9CCFBC',
    fontSize: '0.8rem',
    fontWeight: 500,
  },
  timelineMock: {
    background: '#1D3A32',
    height: '4px',
    borderRadius: '4px',
    flex: 1,
    margin: '0 12px',
  },
  infoText: {
    marginTop: '0.8rem',
    fontSize: '0.75rem',
    color: '#8FBEAE',
    textAlign: 'center',
  },
  actionRow: {
    display: 'flex',
    gap: '12px',
    marginTop: '16px',
    width: '100%',
  },
  cancelBtn: {
    background: 'transparent',
    border: '1.5px solid #E04545',
    padding: '0.7rem 1.6rem',
    borderRadius: '40px',
    fontWeight: 800,
    color: '#E04545',
    fontFamily: 'inherit',
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontSize: '0.9rem',
  },
  predictActionBtn: {
    background: '#142A24',
    border: '1.5px solid #38F0B0',
    padding: '0.7rem 1.6rem',
    borderRadius: '40px',
    fontWeight: 800,
    color: '#EEF5F0',
    fontFamily: 'inherit',
    cursor: 'pointer',
    transition: 'all 0.2s',
    fontSize: '0.9rem',
    flex: 1,
    boxShadow: '0 4px 15px rgba(56, 240, 176, 0.2)',
  },
  iconBtn: {
    background: 'transparent',
    border: '1.5px solid #2DAA7A',
    color: '#C0F5E0',
    width: '48px',
    height: '48px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '50%',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  infoCard: {
    flex: 1.2,
    minWidth: 240,
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  statsPanel: {
    background: '#0A1815',
    borderRadius: '1.25rem',
    padding: '1.3rem',
    border: '1px solid rgba(65, 210, 140, 0.35)',
  },
  statsTitle: {
    fontWeight: 800,
    fontSize: '0.9rem',
    textTransform: 'uppercase',
    color: '#48E0A8',
    marginBottom: '1rem',
    letterSpacing: 1,
  },
  statItem: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '0.8rem',
    borderBottom: '1px dashed rgba(80, 180, 130, 0.3)',
    paddingBottom: '0.6rem',
  },
  statLabel: {
    color: '#AACEC0',
    fontWeight: 500,
  },
  statNumber: {
    color: 'white',
    fontWeight: 800,
    background: '#10231F',
    padding: '0.2rem 0.7rem',
    borderRadius: 20,
    fontFamily: 'monospace',
  },
  deliveryLogSection: {
    marginTop: '1rem',
    paddingTop: '0.8rem',
    borderTop: '1px solid rgba(65, 210, 140, 0.35)',
  },
  deliveryLogTitle: {
    fontWeight: 800,
    fontSize: '0.75rem',
    textTransform: 'uppercase',
    color: '#48E0A8',
    marginBottom: '0.6rem',
    letterSpacing: 0.8,
  },
  deliveryLogList: {
    maxHeight: 220,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    paddingRight: 4,
  },
  deliveryRow: {
    display: 'grid',
    gridTemplateColumns: '36px 1fr 48px',
    alignItems: 'center',
    gap: 8,
    background: '#071212',
    borderRadius: 8,
    padding: '6px 10px',
    border: '1px solid rgba(60, 210, 140, 0.2)',
    fontSize: '0.78rem',
  },
  deliveryNum: {
    color: '#87B7A5',
    fontWeight: 700,
    fontFamily: 'monospace',
  },
  deliveryLength: {
    color: '#EEF5F0',
    fontWeight: 600,
  },
  deliveryOutcome: {
    fontWeight: 800,
    textAlign: 'right',
    fontFamily: 'monospace',
  },
  accentText: {
    color: '#38F0B0',
  },
  eventLogPanel: {
    background: '#0A1815',
  },
  eventLogMsg: {
    fontSize: '0.75rem',
    color: '#B2E0CC',
    fontWeight: 500,
  },
  aiNote: {
    marginTop: 12,
    fontSize: '0.7rem',
    color: '#87B7A5',
  },
  footnote: {
    fontSize: '0.7rem',
    textAlign: 'center',
    marginTop: '1.5rem',
    color: '#7D9F8F',
  },
};

export default CricketTrajectoryPredictor;
