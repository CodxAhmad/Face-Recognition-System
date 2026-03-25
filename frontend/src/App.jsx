import { useState } from "react";
import HealthCheck from "./components/HealthCheck";
import WebcamRecognition from "./components/WebcamRecognition";
import UploadRecognition from "./components/UploadRecognition";
import RegisterPanel from "./components/RegisterPanel";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("recognize");

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect x="2" y="2" width="10" height="10" rx="1" stroke="#00ff88" strokeWidth="1.5"/>
              <rect x="16" y="2" width="10" height="10" rx="1" stroke="#00ff88" strokeWidth="1.5"/>
              <rect x="2" y="16" width="10" height="10" rx="1" stroke="#00ff88" strokeWidth="1.5"/>
              <rect x="16" y="16" width="10" height="10" rx="1" stroke="#00ff88" strokeWidth="1.5"/>
              <circle cx="14" cy="14" r="3" fill="#00ff88"/>
            </svg>
          </div>
          <div>
            <h1 className="app-title">FACE<span>ID</span></h1>
            <p className="app-subtitle">Neural Recognition System</p>
          </div>
        </div>
        <HealthCheck />
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === "recognize" ? "active" : ""}`}
          onClick={() => setActiveTab("recognize")}
        >
          <span className="tab-icon">◉</span> RECOGNIZE
        </button>
        <button
          className={`tab-btn ${activeTab === "register" ? "active" : ""}`}
          onClick={() => setActiveTab("register")}
        >
          <span className="tab-icon">⊕</span> REGISTER
        </button>
      </nav>

      <main className="app-main">
        {activeTab === "recognize" && (
          <div className="recognize-layout">
            <WebcamRecognition />
            <UploadRecognition />
          </div>
        )}
        {activeTab === "register" && (
          <RegisterPanel />
        )}
      </main>
    </div>
  );
}

export default App;