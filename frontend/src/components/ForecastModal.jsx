import { useEffect, useState } from "react";
import { fetchForecast } from "../api/forecast";
import ForecastChart, { SCENARIO_STYLES } from "./ForecastChart";
import ProvenanceBadge from "./ProvenanceBadge";

const MIN_YEARS = 1;
const MAX_YEARS = 15;
const DEFAULT_YEARS = 5;
const DEBOUNCE_MS = 300;

// Builds the chart's series list from whichever shape /forecast returned -
// status is the only field that changes the response shape (see
// ROADMAP.md): "trend"/"undetermined" carry a single forecast array,
// "scenario" carries a named dict of them. Everything downstream (the
// chart, the legend) works from this uniform array either way.
function buildSeries(result) {
  if (!result) return [];
  if (result.status === "scenario") {
    return Object.entries(result.scenarios).map(([name, values]) => ({ key: name, label: name, values }));
  }
  const values = result.status === "undetermined" ? result.trend_forecast : result.forecast;
  return [{ key: "forecast", label: "Forecast", values }];
}

export default function ForecastModal({ country, onClose }) {
  const [yearsAhead, setYearsAhead] = useState(DEFAULT_YEARS);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | error | done
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let cancelled = false;

    // Debounced so dragging the slider doesn't fire a /forecast call for
    // every intermediate value: only the value the user settles on for
    // DEBOUNCE_MS actually triggers a request. The `cancelled` flag
    // separately guards the case where a request is already in flight (past
    // the debounce) when a newer value supersedes it - its response is
    // ignored so slow/out-of-order responses can't clobber a fresher one.
    //
    // Note `result` is deliberately NOT cleared here: the chart keeps
    // showing the last successful result (dimmed via status==="loading")
    // instead of disappearing, so the modal never resizes or blanks out
    // between requests.
    setStatus("loading");
    const timeoutId = setTimeout(() => {
      fetchForecast(country.sipriName, yearsAhead)
        .then((data) => {
          if (cancelled) return;
          setResult(data);
          setStatus("done");
        })
        .catch((err) => {
          if (cancelled) return;
          setErrorMessage(err.message);
          setStatus("error");
        });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [country.sipriName, yearsAhead]);

  const series = buildSeries(result);
  const isScenario = result?.status === "scenario";
  const isUndetermined = result?.status === "undetermined";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(evt) => evt.stopPropagation()}>
        <button className="modal__close" onClick={onClose} aria-label="Close">
          &times;
        </button>

        <h2 className="modal__title">{country.sipriName}</h2>
        <p className="modal__subtitle">
          Most recent known spending:{" "}
          <span className="mono modal__known-value">{(country.latestSpending * 100).toFixed(2)}%</span> of GDP (
          <span className="mono">{country.latestYear}</span>)
        </p>

        {(isScenario || result?.cap_source) && (
          <div className="modal__badges">
            {isScenario && (
              <ProvenanceBadge
                label="Scenarios"
                source={result.scenario_source}
                confidence={result.scenario_confidence}
              />
            )}
            {result?.cap_source && (
              <ProvenanceBadge label="Growth cap" source={result.cap_source} confidence={result.cap_confidence} />
            )}
          </div>
        )}

        {isUndetermined && (
          <p className="modal__undetermined-note">
            <strong>Flagged, not yet researched.</strong> {result.detection_status}
          </p>
        )}

        <label className="modal__slider-label" htmlFor="years-ahead">
          Forecast <span className="mono modal__slider-years">{yearsAhead}</span> year{yearsAhead === 1 ? "" : "s"}{" "}
          ahead
        </label>
        <input
          id="years-ahead"
          type="range"
          min={MIN_YEARS}
          max={MAX_YEARS}
          value={yearsAhead}
          onChange={(evt) => setYearsAhead(Number(evt.target.value))}
          className="modal__slider"
        />

        <div className={`modal__results${isScenario ? " modal__results--scenario" : ""}`}>
          <ForecastChart
            history={country.history}
            series={series}
            latestYear={country.latestYear}
            isLoading={status === "loading"}
          />
          <div className="forecast-chart__legend">
            <span className="forecast-chart__legend-item">
              <span className="forecast-chart__swatch forecast-chart__swatch--observed" /> Observed
            </span>
            {series.map((s, i) => {
              const style = SCENARIO_STYLES[i % SCENARIO_STYLES.length];
              return (
                <span className="forecast-chart__legend-item" key={s.key}>
                  <span
                    className="forecast-chart__swatch forecast-chart__swatch--forecast"
                    style={{
                      background: `repeating-linear-gradient(to right, ${style.color} 0, ${style.color} 4px, transparent 4px, transparent 7px)`,
                    }}
                  />{" "}
                  {s.label}
                </span>
              );
            })}
          </div>
          {isScenario && result.magnitude_note && <p className="modal__magnitude-note">{result.magnitude_note}</p>}
          {status === "error" && (
            <p className="modal__status modal__error modal__error--overlay">
              Failed to load forecast: {errorMessage}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
