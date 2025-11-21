import { useState } from 'react';
import axios from 'axios';
import { Loader2, Send, Download, AlertCircle, Clock } from 'lucide-react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inferenceTime, setInferenceTime] = useState<number | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setImage(null);
    setInferenceTime(null);

    try {
      const response = await axios.post('http://localhost:8000/api/generate', {
        prompt: prompt,
        steps: 30,
        guidance_scale: 7.5
      });

      if (response.data.status === 'success' && response.data.image_base64) {
        setImage(`data:image/png;base64,${response.data.image_base64}`);
        if (response.data.inference_time) {
          setInferenceTime(response.data.inference_time);
        }
      } else {
        setError(response.data.error || 'Unknown error occurred');
      }
    } catch (err) {
      setError('Failed to connect to the server. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-white selection:text-black">
      <div className="container mx-auto px-4 py-16 max-w-5xl">
        {/* Header */}
        <header className="mb-16 text-center space-y-4">
          <h1 className="text-6xl font-bold tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent">
            MediSynVision
          </h1>
          <p className="text-zinc-400 text-xl font-light tracking-wide">
            Advanced Synthetic Medical Imaging
          </p>
        </header>

        {/* Main Interface */}
        <main className="space-y-12">
          {/* Input Section */}
          <div className="bg-zinc-900/50 p-2 rounded-full shadow-2xl shadow-black/50 border border-zinc-800 backdrop-blur-sm transition-all focus-within:border-zinc-600 focus-within:ring-1 focus-within:ring-zinc-600">
            <div className="relative flex items-center">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
                placeholder="Describe the pathology to generate..."
                className="w-full bg-transparent border-none rounded-full px-8 py-4 text-lg focus:outline-none focus:ring-0 placeholder-zinc-600 text-white"
                disabled={loading}
              />
              <button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className="absolute right-2 bg-white hover:bg-zinc-200 text-black font-medium px-8 py-3 rounded-full transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-white/10 hover:shadow-white/20 hover:scale-105 active:scale-95"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
                Generate
              </button>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-950/30 border border-red-900/50 text-red-200 px-6 py-4 rounded-2xl flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
              <AlertCircle className="w-5 h-5" />
              {error}
            </div>
          )}

          {/* Result Display */}
          <div className="min-h-[500px] flex items-center justify-center bg-zinc-900/30 rounded-[2.5rem] border border-zinc-800/50 shadow-inner relative overflow-hidden group">
            {loading ? (
              <div className="text-center space-y-6">
                <div className="relative">
                  <div className="absolute inset-0 bg-white/20 blur-xl rounded-full animate-pulse"></div>
                  <Loader2 className="w-16 h-16 animate-spin text-white relative z-10" />
                </div>
                <p className="text-zinc-500 font-light tracking-widest uppercase text-sm animate-pulse">Processing</p>
              </div>
            ) : image ? (
              <div className="relative w-full h-full flex flex-col items-center p-8 animate-in zoom-in-95 duration-500">
                <div className="relative rounded-2xl overflow-hidden shadow-2xl shadow-black/80 ring-1 ring-white/10">
                  <img
                    src={image}
                    alt="Generated medical scan"
                    className="max-w-full h-auto max-h-[600px] object-contain"
                  />

                  {/* Inference Time Badge */}
                  {inferenceTime && (
                    <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md text-white/90 px-4 py-1.5 rounded-full text-xs font-medium border border-white/10 flex items-center gap-2 shadow-lg">
                      <Clock className="w-3 h-3" />
                      {inferenceTime.toFixed(2)}s
                    </div>
                  )}

                  {/* Download Overlay */}
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center backdrop-blur-[2px]">
                    <a
                      href={image}
                      download={`medisynvision-${Date.now()}.png`}
                      className="bg-white text-black px-8 py-4 rounded-full font-semibold flex items-center gap-3 hover:scale-105 transition-transform shadow-xl hover:bg-zinc-100"
                    >
                      <Download className="w-5 h-5" />
                      Save Image
                    </a>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-zinc-700 text-center space-y-4">
                <div className="w-24 h-24 rounded-full bg-zinc-900/50 border border-zinc-800 flex items-center justify-center mx-auto">
                  <Send className="w-8 h-8 text-zinc-800" />
                </div>
                <p className="text-lg font-light">Ready to generate synthetic imagery</p>
              </div>
            )}
          </div>
        </main>

        <footer className="mt-24 text-center text-zinc-600 text-sm">
          <p>© 2025 MediSynVision Research. Powered by Stable Diffusion.</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
