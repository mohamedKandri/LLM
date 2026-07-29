import { useState } from "react";
import "./App.css";
import { Dashboard } from "./components/Dashboard";
import { SettingsPanel } from "./components/SettingsPanel";
import { SeedEditor } from "./components/SeedEditor";

type Tab = "dashboard" | "settings" | "seeds";

function App() {
  const [tab, setTab] = useState<Tab>("dashboard");

  return (
    <div className="app">
      <header className="app-header">
        <h1>Distill</h1>
        <nav className="tabs">
          {(["dashboard", "settings", "seeds"] as const).map((t) => (
            <button
              key={t}
              className={`tab-button ${tab === t ? "active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
      </header>
      <main className="app-body">
        {tab === "dashboard" && <Dashboard />}
        {tab === "settings" && <SettingsPanel />}
        {tab === "seeds" && <SeedEditor />}
      </main>
    </div>
  );
}

export default App;
