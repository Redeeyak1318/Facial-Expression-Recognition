'use client';

import { useState, useRef } from 'react';
import { predictImage, PredictionResult } from '@/lib/api';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [imageNativeSize, setImageNativeSize] = useState<{width: number, height: number} | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setError(null);
      setResult(null);
      setImageNativeSize(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      if (selectedFile.type.startsWith('image/')) {
        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
        setError(null);
        setResult(null);
        setImageNativeSize(null);
      } else {
        setError('Please drop a valid image file.');
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err: any) {
      if (err.message === 'NO_FACE_DETECTED') {
        setError('No face detected. Please upload a clearer photo containing a visible face.');
      } else {
        setError(err.message || 'An error occurred during prediction.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setImageNativeSize({
      width: e.currentTarget.naturalWidth,
      height: e.currentTarget.naturalHeight
    });
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setImageNativeSize(null);
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4 font-sans text-gray-900">
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-xl overflow-hidden flex flex-col">
        <div className="bg-indigo-600 p-6 text-center">
          <h1 className="text-3xl font-extrabold text-white">Facial Expression Recognition</h1>
          <p className="text-indigo-100 mt-2">Upload a photo to detect facial expression.</p>
        </div>

        <div className="p-8 flex-1 flex flex-col">
          {!preview && (
            <div 
              className="border-2 border-dashed border-gray-300 rounded-xl p-16 text-center cursor-pointer hover:bg-gray-50 hover:border-indigo-400 transition-colors"
              onClick={handleUploadClick}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                className="hidden" 
                accept="image/*" 
              />
              <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="mt-6 text-xl font-medium text-gray-900">Upload Image</p>
              <p className="mt-2 text-md text-gray-500">or drag & drop</p>
            </div>
          )}

          {preview && !result && (
            <div className="flex flex-col items-center">
              <div className="relative rounded-lg overflow-hidden shadow-md max-w-md w-full bg-gray-100">
                <img src={preview} alt="Preview" className="w-full h-auto block" onLoad={handleImageLoad} />
              </div>
              
              {error && (
                <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 w-full max-w-md">
                  <p className="text-red-700 text-center font-medium">{error}</p>
                </div>
              )}

              <div className="mt-8 flex gap-4 w-full max-w-md">
                <button 
                  onClick={reset}
                  className="flex-1 py-3 px-4 border border-gray-300 rounded-lg text-gray-700 font-bold hover:bg-gray-50 transition-colors"
                  disabled={loading}
                >
                  Cancel
                </button>
                <button 
                  onClick={handleAnalyze}
                  className="flex-1 py-3 px-4 bg-indigo-600 rounded-lg text-white font-bold hover:bg-indigo-700 transition-colors disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center"
                  disabled={loading}
                >
                  {loading ? (
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : 'Analyze Expression'}
                </button>
              </div>
            </div>
          )}

          {result && (
            <div className="flex flex-col">
              <div className="text-center mb-10">
                <p className="text-sm text-gray-500 uppercase tracking-wider font-semibold">Predicted Expression</p>
                <h2 className="text-6xl font-black text-indigo-600 mt-2 uppercase">{result.predicted_class}</h2>
                <p className="text-2xl text-gray-700 mt-3">
                  Confidence: <span className="font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                <div>
                  <h3 className="text-xl font-bold text-gray-900 border-b pb-3 mb-5">Probability Distribution</h3>
                  <div className="space-y-4">
                    {Object.entries(result.probabilities).map(([expression, prob]) => (
                      <div key={expression}>
                        <div className="flex justify-between text-md mb-1.5">
                          <span className="font-semibold capitalize text-gray-800">{expression}</span>
                          <span className="text-gray-600">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className={`h-3 rounded-full ${expression === result.predicted_class ? 'bg-indigo-600' : 'bg-gray-400'}`} 
                            style={{ width: `${Math.max(prob * 100, 1)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex flex-col">
                  <h3 className="text-xl font-bold text-gray-900 border-b pb-3 mb-5">Detection Details</h3>
                  <div className="relative rounded-xl overflow-hidden bg-gray-100 flex-1 flex items-center justify-center border border-gray-200 p-2 shadow-inner">
                    <div className="relative inline-block">
                      <img src={preview!} alt="Original" className="max-w-full max-h-[280px] block rounded-lg shadow" onLoad={handleImageLoad} />
                      {result.bbox && imageNativeSize && (
                        <div 
                          className="absolute border-4 border-green-500 rounded-sm"
                          style={{
                            left: `${(result.bbox[0] / imageNativeSize.width) * 100}%`,
                            top: `${(result.bbox[1] / imageNativeSize.height) * 100}%`,
                            width: `${(result.bbox[2] / imageNativeSize.width) * 100}%`,
                            height: `${(result.bbox[3] / imageNativeSize.height) * 100}%`,
                            boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5)'
                          }}
                        />
                      )}
                    </div>
                  </div>
                  <div className="mt-4 text-md text-gray-700 bg-indigo-50 p-4 rounded-lg border border-indigo-100">
                    <span className="font-bold text-indigo-900">Face Bounding Box:</span><br/>
                    <div className="grid grid-cols-2 gap-2 mt-2 font-mono text-sm">
                      <div><span className="text-gray-500">x:</span> {result.bbox[0]}</div>
                      <div><span className="text-gray-500">y:</span> {result.bbox[1]}</div>
                      <div><span className="text-gray-500">width:</span> {result.bbox[2]}</div>
                      <div><span className="text-gray-500">height:</span> {result.bbox[3]}</div>
                    </div>
                    <p className="mt-3 text-xs text-indigo-600 font-semibold">* Using the largest detected face.</p>
                  </div>
                </div>
              </div>

              <div className="mt-12 text-center">
                <button 
                  onClick={reset}
                  className="py-4 px-10 bg-gray-100 hover:bg-gray-200 text-gray-800 font-bold rounded-xl transition-colors border border-gray-300 shadow-sm"
                >
                  Try another image
                </button>
              </div>
            </div>
          )}
        </div>
        
        <div className="bg-gray-100 p-4 text-center border-t border-gray-200">
          <p className="text-xs text-gray-500">Expression prediction is an AI estimate and may be inaccurate.</p>
        </div>
      </div>
    </main>
  );
}
