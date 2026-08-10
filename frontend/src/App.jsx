import { useState } from "react";
import WorldMap from "./components/WorldMap";
import ComparisonPanel from "./components/ComparisonPanel";
import InfoPanel from "./components/InfoPanel";
import FactTicker from "./components/FactTicker";
import TypedText from "./components/TypedText";
import { countryDataByMapName, datasetYearRange } from "./data/countryData";
import "./App.css";

const COUNTRY_COUNT = countryDataByMapName.size;

const TITLE = "Military Spending Forecast Tool";
const SUBTITLE =
  "31 countries, 46 years of SIPRI spending data. Trend-based forecasts, with named scenarios for " +
  "countries where a single trend line wouldn’t be honest.";

// Ukraine starts pinned: our best, most validated result (a real
// ground-truth blind test, genuine scenario divergence - see
// docs/ROADMAP_v2_working_notes.md), visible on load rather than something
// a visitor has to discover by clicking. Found by sipriName rather than
// hardcoding a map-name lookup, since countryDataByMapName is keyed by the
// map's own naming (see data/countryData.js).
const DEFAULT_PINNED = [...countryDataByMapName.values()].filter((c) => c.sipriName === "Ukraine");

function App() {
  const [pinnedCountries, setPinnedCountries] = useState(DEFAULT_PINNED);
  const [isComparisonOpen, setIsComparisonOpen] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  // Gates the subtitle's TypedText - it only starts once the title's own
  // TypedText reports done, so the two type out sequentially (title, then
  // subtitle) rather than both at once.
  const [titleTyped, setTitleTyped] = useState(false);

  function handleAddCountry(country) {
    setPinnedCountries((prev) =>
      prev.some((c) => c.sipriName === country.sipriName) ? prev : [...prev, country]
    );
    // A click that silently updates state behind a closed panel would look
    // like nothing happened - surfacing the panel is what makes the click
    // itself legible as "added," not just a decision made in the abstract.
    setIsComparisonOpen(true);
  }

  function handleRemoveCountry(sipriName) {
    setPinnedCountries((prev) => prev.filter((c) => c.sipriName !== sipriName));
  }

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__masthead">
          <h1 aria-label={TITLE}>
            <TypedText text={TITLE} charIntervalMs={65} onDone={() => setTitleTyped(true)} />
          </h1>
          {/* Verified live against backend/model_selection_log.json, not
              the (now-stale) 27/31 figure in README.md - see
              docs/ROADMAP_v2_working_notes.md's yearly_seasonality fix,
              which flipped 4 more countries to Prophet after that figure
              was originally written. */}
          <div className="app__masthead-stat">
            <span className="app__masthead-stat-value mono">30 / 31</span>
            <span className="app__masthead-stat-label">Prophet&rsquo;s win rate</span>
          </div>
        </div>
        {/* Editorial deck line, not UI instructions - see App.css for the
            hierarchy this is deliberately secondary to. The "how to use
            this" mechanics moved to data-strip__hint below, near the
            legend they actually explain. Types out only after the title
            finishes (active={titleTyped}), so the two read as one
            sequential effect rather than both typing at once. Faster per
            character than the title (20ms vs. 65ms) - at the title's own
            pace this 150-character line would take ~10s alone, which
            reads as theatrical, not "fast and subtle." */}
        <p aria-label={SUBTITLE}>
          <TypedText text={SUBTITLE} charIntervalMs={20} active={titleTyped} />
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
          <span className="data-strip__hint">
            Hover a country for its latest value &middot; click to add it to the comparison panel
          </span>
        </div>
        <div className="data-strip__stat">
          <span className="mono">{COUNTRY_COUNT}</span> countries &middot;{" "}
          <span className="mono">
            {datasetYearRange.earliest}&ndash;{datasetYearRange.latest}
          </span>
        </div>
      </div>

      <WorldMap onSelectCountry={handleAddCountry} />

      <FactTicker />

      <ComparisonPanel
        pinnedCountries={pinnedCountries}
        onRemove={handleRemoveCountry}
        isOpen={isComparisonOpen}
        onToggle={() => setIsComparisonOpen((open) => !open)}
        onClose={() => setIsComparisonOpen(false)}
      />

      <InfoPanel
        isOpen={isInfoOpen}
        onToggle={() => setIsInfoOpen((open) => !open)}
        onClose={() => setIsInfoOpen(false)}
      />
    </div>
  );
}

export default App;
