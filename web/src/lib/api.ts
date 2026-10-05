export interface PredictionResult {
  success: boolean;
  predicted_class: string;
  confidence: number;
  probabilities: {
    angry: number;
    disgust: number;
    fear: number;
    happy: number;
    neutral: number;
    sad: number;
    surprise: number;
  };
  bbox: [number, number, number, number];
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${apiUrl}/predict`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    if (response.status === 422) {
      const errorData = await response.json();
      if (errorData.error === 'NO_FACE_DETECTED') {
        throw new Error('NO_FACE_DETECTED');
      }
    }
    throw new Error('Failed to analyze image. Please ensure the backend is running.');
  }
  
  return response.json();
}
