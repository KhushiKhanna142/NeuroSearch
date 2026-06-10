import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, ImageIcon } from 'lucide-react';
import { SectionHeader, ConfidenceBar } from '../components';
import { api } from '../api';

const CIFAR10_CLASSES = [
  'Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck'
];
const CLASS_EMOJIS = ['✈️','🚗','🐦','🐱','🦌','🐶','🐸','🐴','🚢','🚛'];

export default function InferenceDemo() {
  const [imageURL, setImageURL]   = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [finetuneData, setFinetuneData] = useState(null);

  React.useEffect(() => {
    api.getFinetuneResults()
      .then(r => setFinetuneData(r.data))
      .catch(() => {});
  }, []);

  const loadImage = (file) => {
    setImageFile(file);
    setImageURL(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const onDrop = useCallback(files => {
    if (files[0]) loadImage(files[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': ['.png', '.jpg', '.jpeg'] }, multiple: false,
  });

  const handleSample = (classIdx) => {
    // Generate a simple coloured placeholder canvas for the selected class
    const canvas = document.createElement('canvas');
    canvas.width = 32; canvas.height = 32;
    const ctx = canvas.getContext('2d');
    const colours = ['#3B82F6','#EF4444','#10B981','#F59E0B','#6B7280',
                     '#8B5CF6','#EC4899','#F97316','#06B6D4','#84CC16'];
    ctx.fillStyle = colours[classIdx];
    ctx.fillRect(0, 0, 32, 32);
    ctx.fillStyle = '#fff';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(CLASS_EMOJIS[classIdx], 16, 22);
    canvas.toBlob(blob => loadImage(new File([blob], `${CIFAR10_CLASSES[classIdx]}.png`)));
  };

  const classify = async () => {
    if (!imageFile) return;
    setLoading(true); setError(null);
    const fd = new FormData();
    fd.append('file', imageFile);
    try {
      const r = await api.runInference(fd);
      setResult(r.data);
    } catch (e) {
      setError(e.response?.data?.detail || 'Inference failed — is the backend running?');
    }
    setLoading(false);
  };

  return (
    <div>
      <SectionHeader title="Inference Demo"
        subtitle="Upload any image or pick a CIFAR-10 sample — the NAS-found model classifies it instantly" />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>
        {/* Input Panel */}
        <div>
          {/* Drop zone */}
          <div {...getRootProps()} style={{
            border: `2px dashed ${isDragActive ? '#2E75B6' : '#CBD5E1'}`,
            borderRadius: 12, padding: 32, textAlign: 'center', cursor: 'pointer',
            background: isDragActive ? '#EFF6FF' : '#F8FAFC',
            transition: 'all 0.2s', marginBottom: 16,
          }}>
            <input {...getInputProps()} />
            {imageURL ? (
              <div>
                <img src={imageURL} alt="uploaded"
                  style={{ maxHeight: 200, maxWidth: '100%', borderRadius: 8, marginBottom: 12,
                    imageRendering: 'pixelated', border: '1px solid #E2E8F0' }} />
                <div style={{ fontSize: 12, color: '#64748B' }}>Click or drop to replace</div>
              </div>
            ) : (
              <>
                <Upload size={36} color="#94A3B8" style={{ marginBottom: 12 }} />
                <div style={{ fontSize: 15, fontWeight: 600, color: '#475569', marginBottom: 6 }}>
                  {isDragActive ? 'Drop it!' : 'Drag & drop an image here'}
                </div>
                <div style={{ fontSize: 12, color: '#94A3B8' }}>PNG or JPG accepted</div>
              </>
            )}
          </div>

          {/* Sample picker */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: '#64748B', marginBottom: 8, fontWeight: 500 }}>
              Or pick a CIFAR-10 class sample:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {CIFAR10_CLASSES.map((cls, i) => (
                <button key={i} onClick={() => handleSample(i)} style={{
                  padding: '5px 10px', fontSize: 12, borderRadius: 20,
                  border: '1px solid #E2E8F0', background: '#fff', cursor: 'pointer',
                  color: '#475569', display: 'flex', alignItems: 'center', gap: 4,
                  transition: 'all 0.15s',
                }}
                  onMouseOver={e => e.currentTarget.style.borderColor = '#2E75B6'}
                  onMouseOut={e => e.currentTarget.style.borderColor = '#E2E8F0'}
                >
                  {CLASS_EMOJIS[i]} {cls}
                </button>
              ))}
            </div>
          </div>

          {/* Caption */}
          <div style={{ fontSize: 11, color: '#94A3B8', marginBottom: 16 }}>
            Images are resized to 32×32 and normalised before inference.
          </div>

          {/* Classify button */}
          <button onClick={classify} disabled={!imageFile || loading} style={{
            width: '100%', padding: '12px 0', fontSize: 15, fontWeight: 700,
            borderRadius: 10, border: 'none', cursor: imageFile ? 'pointer' : 'not-allowed',
            background: imageFile ? '#2E75B6' : '#E2E8F0',
            color: imageFile ? '#fff' : '#94A3B8', transition: 'all 0.2s',
            boxShadow: imageFile ? '0 2px 8px rgba(46,117,182,0.3)' : 'none',
          }}>
            {loading ? 'Classifying…' : 'Classify'}
          </button>

          {error && (
            <div style={{
              marginTop: 12, background: '#FEF2F2', border: '1px solid #FECACA',
              borderRadius: 8, padding: '10px 14px', color: '#DC2626', fontSize: 13,
            }}>⚠️ {error}</div>
          )}
        </div>

        {/* Results Panel */}
        <div>
          {result ? (
            <div style={{
              background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
              padding: 24, boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
            }}>
              {/* Top Prediction */}
              <div style={{
                background: 'linear-gradient(135deg, #1E3A5F, #2E75B6)',
                borderRadius: 10, padding: '20px 24px', marginBottom: 20, textAlign: 'center',
              }}>
                <div style={{ fontSize: 40, marginBottom: 6 }}>
                  {CLASS_EMOJIS[CIFAR10_CLASSES.indexOf(result.top_class)] || '🤖'}
                </div>
                <div style={{ fontSize: 28, fontWeight: 800, color: '#fff', letterSpacing: '-0.5px' }}>
                  {result.top_class}
                </div>
                <div style={{ fontSize: 22, color: '#93C5FD', fontWeight: 700, marginTop: 4 }}>
                  {(result.top_confidence * 100).toFixed(1)}%
                </div>
              </div>

              {/* Top-5 bars */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 12 }}>Top-5 Predictions</div>
                {result.top5.map((c, i) => (
                  <ConfidenceBar key={c.label} label={c.label} value={c.confidence} isTop={i === 0} />
                ))}
              </div>

              {/* Footer info */}
              <div style={{
                borderTop: '1px solid #F1F5F9', paddingTop: 12, marginTop: 8,
                display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94A3B8',
              }}>
                <span>⚡ {result.latency_ms.toFixed(1)} ms inference</span>
                <span>{finetuneData && finetuneData.test_top1 
                  ? `Fine-tuned Model (${(finetuneData.test_top1*100).toFixed(1)}% Acc, ${(finetuneData.params/1e3).toFixed(0)}K params)` 
                  : "NAS-found model · ONNX Runtime"}</span>
              </div>
            </div>
          ) : (
            <div style={{
              background: '#F8FAFC', border: '2px dashed #E2E8F0', borderRadius: 12,
              padding: 48, textAlign: 'center', color: '#94A3B8',
            }}>
              <ImageIcon size={48} style={{ marginBottom: 14, opacity: 0.4 }} />
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>Results will appear here</div>
              <div style={{ fontSize: 13 }}>Upload an image and click Classify</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
