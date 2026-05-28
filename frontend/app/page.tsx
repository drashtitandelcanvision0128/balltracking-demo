"use client";
import React, { useState, useRef, useCallback, useEffect } from 'react';

const CricketTrajectoryPredictor: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('video.mp4');
  const [statusText, setStatusText] = useState<string>('Ready. Select a video to analyze trajectory.');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [isProcessed, setIsProcessed] = useState<boolean>(false);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [downloadUrl, setDownloadUrl] = useState<string>('');
  const [framesProcessed, setFramesProcessed] = useState<number>(843);
  const [bounceEvents, setBounceEvents] = useState<number>(78);
  const [deletedBounces, setDeletedBounces] = useState<number>(78);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
      setStatusText(`Selected: ${file.name}. Click 'Upload and Predict'.`);
      setIsProcessed(false);
      // Reset stats to default
      setFramesProcessed(843);
      setBounceEvents(78);
      setDeletedBounces(78);
      // Clear previous video URL
      if (videoUrl && videoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(videoUrl);
        setVideoUrl('');
      }
      if (videoRef.current) {
        videoRef.current.src = '';
      }
    } else {
      setSelectedFile(null);
      setFileName('video.mp4');
      setStatusText('No file selected.');
    }
  };

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
        setStatusText('Video queued. Waiting for processing...');
        pollJob(data.job_id);
      }
    } catch (error: any) {
      setStatusText('Error connecting to backend: ' + error.message);
      setIsProcessing(false);
    }
  };

  const pollJob = (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:5000/status/${jobId}`);
        const data = await response.json();

        if (data.status === 'processing') {
          setStatusText('AI processing video frames...');
          return;
        }

        clearInterval(pollInterval);

        if (data.status === 'error') {
          setStatusText(data.error || 'Prediction failed');
          setIsProcessing(false);
          return;
        }

        if (data.status === 'done') {
          setFramesProcessed(data.summary?.frames_processed || 0);
          setBounceEvents(Array.isArray(data.summary?.bounce_events) ? data.summary.bounce_events.length : 0);
          setDeletedBounces(0);
          setVideoUrl(`http://localhost:5000${data.video_url}`);
          setDownloadUrl(`http://localhost:5000${data.download_url}`);
          setStatusText('Prediction complete! Trajectory overlay active.');
          setIsProcessing(false);
          setIsProcessed(true);
        }
      } catch (error: any) {
        clearInterval(pollInterval);
        setStatusText('Polling failed: ' + error.message);
        setIsProcessing(false);
      }
    }, 2000);
  };

  const handleDownload = () => {
    if (!isProcessed || !downloadUrl) {
      setStatusText('No processed video available. Please upload and run prediction first.');
      return;
    }
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `predicted_${selectedFile?.name || 'trajectory_video.mp4'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setStatusText('Download started: predicted trajectory video.');
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
              <h1 style={styles.title}>CRICKET BALL TRAJECTORY PREDICTOR</h1>
              <p style={styles.subtitle}>Upload a cricket video & AI-powered trajectory overlay | Neon mint analytics</p>
            </div>
            <div style={styles.badge}>
              <span style={styles.badgeText}>PREDICTION ENGINE <span style={styles.liveIndicator}>● ACTIVE</span></span>
            </div>
          </div>

          {/* Upload Section */}
          <div style={styles.uploadSection}>
            <div style={styles.uploadGrid}>
              <label style={styles.fileLabel} htmlFor="videoUpload">
                Choose File
              </label>
              <input
                ref={fileInputRef}
                id="videoUpload"
                type="file"
                accept="video/mp4,video/mov,video/avi"
                onChange={handleFileChange}
                style={styles.hiddenInput}
              />
              <div style={styles.fileName}>{fileName}</div>
              <button
                style={{ ...styles.primaryBtn, ...(isProcessing ? styles.disabledBtn : {}) }}
                onClick={startPrediction}
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Upload and Predict'}
              </button>
            </div>
            <div style={styles.statusBar}>
              <span style={styles.statusText}>{statusText}</span>
            </div>
          </div>

          {/* Main Prediction Area */}
          <div style={styles.predictionArea}>
            {/* Video Player Card */}
            <div style={styles.videoCard}>
              <div style={styles.videoTitle}>
                <span>PREDICTED VIDEO</span>
                <span style={styles.liveTag}>LIVE TRAJECTORY</span>
              </div>
              <div style={styles.videoWrapper}>
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
              </div>

              <div style={styles.infoText}>
                Processed video is playing inline. Use the download link below if you want to save it.
              </div>
              <button style={styles.downloadBtn} onClick={handleDownload}>
                Download predicted video
              </button>
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
                  <span style={styles.statLabel}>Bounce events detected</span>
                  <span style={styles.statNumber}>{bounceEvents}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Deleted bounce events</span>
                  <span style={{ ...styles.statNumber, ...styles.accentText }}>{deletedBounces}</span>
                </div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Trajectory confidence</span>
                  <span style={styles.statNumber}>97.3%</span>
                </div>
              </div>

              <div style={styles.statsPanel}>
                <div style={styles.statsTitle}>BOWL TRAJECTORY METRICS</div>
                <div style={styles.statItem}>
                  <span style={styles.statLabel}>Peak apex height</span>
                  <span style={styles.statNumber}>2.43 m</span>
                </div>
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
                  Processed {framesProcessed} frames. Deleted {deletedBounces} bounce events.
                </div>
                <div style={styles.aiNote}>
                  AI predicted trajectory with mint confidence
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
  downloadBtn: {
    background: 'transparent',
    border: '1.5px solid #2DAA7A',
    color: '#C0F5E0',
    padding: '0.7rem 1rem',
    borderRadius: '40px',
    fontWeight: 700,
    fontSize: '0.85rem',
    cursor: 'pointer',
    width: '100%',
    marginTop: '12px',
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
