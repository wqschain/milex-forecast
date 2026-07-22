import { useState } from "react";
import WorldMap from "./components/WorldMap";
import ForecastModal from "./components/ForecastModal";
import "./App.css";

function App() {
  const [selectedCountry, setSelectedCountry] = useState(null);

  return (
    <div className="app">
      <header className="app__header">
        <h1>Military Spending Forecast</h1>
        <p>
          Highlighted countries have SIPRI historical data and a trained forecasting model. Hover for the most
          recent known value, click to forecast ahead.
        </p>
      </header>

      <WorldMap onSelectCountry={setSelectedCountry} />

      {selectedCountry && (
        <ForecastModal country={selectedCountry} onClose={() => setSelectedCountry(null)} />
      )}
    </div>
  );
}

export default App;
