import { useState } from "react";
import WorldMap from "./components/WorldMap";
import ForecastModal from "./components/ForecastModal";
import FactTicker from "./components/FactTicker";
import { countryDataByMapName, datasetYearRange } from "./data/countryData";
import "./App.css";

const COUNTRY_COUNT = countryDataByMapName.size;

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

      <div className="data-strip">
        <div className="data-strip__legend">
          <span className="legend-item">
            <span className="legend-swatch legend-swatch--data" />
            Has forecast model
          </span>
          <span className="legend-item">
            <span className="legend-swatch legend-swatch--no-data" />
            No data
          </span>
        </div>
        <div className="data-strip__stat">
          <span className="mono">{COUNTRY_COUNT}</span> countries &middot;{" "}
          <span className="mono">
            {datasetYearRange.earliest}&ndash;{datasetYearRange.latest}
          </span>
        </div>
      </div>

      <WorldMap onSelectCountry={setSelectedCountry} />

      <FactTicker />

      {selectedCountry && (
        <ForecastModal country={selectedCountry} onClose={() => setSelectedCountry(null)} />
      )}
    </div>
  );
}

export default App;
