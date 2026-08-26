import { useEffect, useState } from "react";
import { api } from "./api";
import PipelinePanel from "./components/pipeline";
import SlideViewer from "./components/viewer";
import "./App.css";

function SlidePicker({ slideId, onChange }) {
  const [slides, setSlides] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .listSlides()
      .then(setSlides)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="slide-picker">
      <label htmlFor="slide-select">Slide</label>
      <select
        id="slide-select"
        value={slideId}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Select a slide…</option>
        {slides.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
      {error && <span className="slide-picker-error">Could not load slide list: {error}</span>}
    </div>
  );
}

function App() {
  const [slideId, setSlideId] = useState("");

  return (
    <div className="app-shell">
      <header className="app-topbar">
        <h1>HPC Tile Explorer</h1>
        <span className="app-topbar-note">
          JS UI (preview) — talks to the same backend as app_v28.py, which keeps running unchanged.
        </span>
      </header>

      <div className="app-body">
        <aside className="app-sidebar">
          <PipelinePanel />
        </aside>

        <main className="app-main">
          <SlidePicker slideId={slideId} onChange={setSlideId} />
          <SlideViewer slideId={slideId} />
        </main>
      </div>
    </div>
  );
}

export default App;
