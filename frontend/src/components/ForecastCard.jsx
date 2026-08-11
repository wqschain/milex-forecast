import { Fragment, useEffect, useState } from "react";
import { fetchForecast } from "../api/forecast";
import ForecastChart, { SCENARIO_STYLES } from "./ForecastChart";
import ProvenanceBadge from "./ProvenanceBadge";

const MIN_YEARS = 1;
const MAX_YEARS = 15;
const DEFAULT_YEARS = 5;
const DEBOUNCE_MS = 300;

// Builds the chart's series list from whichever shape /forecast returned -
// status is the only field that changes the response shape (see
// docs/ROADMAP_v2_working_notes.md): "trend"/"undetermined" carry a single forecast array,
// "scenario" carries a named dict of them. Everything downstream (the
// chart, the legend) works from this uniform array either way.
//
// forecast_lower/forecast_upper (Prophet's own prediction interval) are
// attached to the single series whenever the backend sends them - which is
// only for status="trend" (see main.py). Gated on the fields' presence, not
// on status or country name, so this keeps working unchanged for any future
// trend-only country (gets a band automatically) or any future scenario
// country, hand-built or automated (never gets one, since scenario
// responses never carry these fields - the scenarios themselves already
// represent the uncertainty).
function buildSeries(result) {
  if (!result) return [];
  if (result.status === "scenario") {
    return Object.entries(result.scenarios).map(([name, values]) => ({ key: name, label: name, values }));
  }
  const values = result.status === "undetermined" ? result.trend_forecast : result.forecast;
  const series = { key: "forecast", label: "Forecast", values };
  if (result.forecast_lower && result.forecast_upper) {
    series.lower = result.forecast_lower;
    series.upper = result.forecast_upper;
  }
  return [series];
}

/**
 * One pinned country's forecast, as a card inside ComparisonPanel - the
 * same fetch/render logic the single-country modal used to own, just
 * without the modal chrome (backdrop, close-to-dismiss), so multiple of
 * these can sit side by side and each fetches/re-renders independently of
 * its neighbors. `years_ahead` is deliberately per-card, not shared across
 * the panel, matching how it already worked before the panel existed - the
 * panel hosts these, it doesn't change how any one of them behaves.
 */
export default function ForecastCard({ country, onRemove }) {
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
    // instead of disappearing, so the card never resizes or blanks out
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
  // First-ever fetch for this card (never had a successful result) is
  // handled as its own state, replacing the chart entirely - a dimmed,
  // nearly-empty chart with no forecast series yet reads as broken/frozen,
  // not "loading." Once a result has landed at least once, a refetch
  // (slider change) instead keeps showing that last-good chart (dimmed)
  // plus a small "Updating..." cue, so the card never blanks out again
  // after its first successful load.
  const hasNeverLoaded = !result;

  return (
    <div className="forecast-card">
      <button className="forecast-card__remove" onClick={onRemove} aria-label={`Remove ${country.sipriName}`}>
        &times;
      </button>

      <h2 className="forecast-card__title">{country.sipriName}</h2>
      <p className="forecast-card__subtitle">
        Most recent known spending:{" "}
        <span className="mono forecast-card__known-value">{(country.latestSpending * 100).toFixed(2)}%</span> of GDP
        (<span className="mono">{country.latestYear}</span>)
      </p>

      {(isScenario || result?.cap_source) && (
        <div className="forecast-card__badges">
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
        <p className="forecast-card__undetermined-note">
          <strong>Flagged, not yet researched.</strong> {result.detection_status}
        </p>
      )}

      <label className="forecast-card__slider-label" htmlFor={`years-ahead-${country.sipriName}`}>
        Forecast <span className="mono forecast-card__slider-years">{yearsAhead}</span> year
        {yearsAhead === 1 ? "" : "s"} ahead
        {!hasNeverLoaded && status === "loading" && (
          <span className="forecast-card__updating">
            <span className="forecast-card__spinner" aria-hidden="true" /> Updating&hellip;
          </span>
        )}
      </label>
      <input
        id={`years-ahead-${country.sipriName}`}
        type="range"
        min={MIN_YEARS}
        max={MAX_YEARS}
        value={yearsAhead}
        onChange={(evt) => setYearsAhead(Number(evt.target.value))}
        className="forecast-card__slider"
      />

      <div className={`forecast-card__results${isScenario ? " forecast-card__results--scenario" : ""}`}>
        {hasNeverLoaded ? (
          status === "error" ? (
            <div className="forecast-card__error-state" role="alert">
              <p>
                <strong>Couldn&rsquo;t load a forecast for {country.sipriName}.</strong>
              </p>
              <p className="forecast-card__error-detail">{errorMessage}</p>
            </div>
          ) : (
            <div className="forecast-card__loading-state" aria-live="polite">
              <span className="forecast-card__spinner forecast-card__spinner--large" aria-hidden="true" />
              <p>Loading forecast&hellip;</p>
            </div>
          )
        ) : (
          <>
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
                  <Fragment key={s.key}>
                    <span className="forecast-chart__legend-item">
                      <span
                        className="forecast-chart__swatch forecast-chart__swatch--forecast"
                        style={{
                          background: `repeating-linear-gradient(to right, ${style.color} 0, ${style.color} 4px, transparent 4px, transparent 7px)`,
                        }}
                      />{" "}
                      {s.label}
                    </span>
                    {s.lower && (
                      <span className="forecast-chart__legend-item">
                        <span
                          className="forecast-chart__swatch forecast-chart__swatch--band"
                          style={{ backgroundColor: style.color }}
                        />{" "}
                        Confidence interval
                      </span>
                    )}
                  </Fragment>
                );
              })}
            </div>
            {isScenario && result.magnitude_note && (
              <p className="forecast-card__magnitude-note">{result.magnitude_note}</p>
            )}
            {status === "error" && (
              <p className="forecast-card__status forecast-card__error forecast-card__error--overlay" role="alert">
                Showing the last successful result &mdash; couldn&rsquo;t refresh: {errorMessage}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
