import { useEffect, useState } from "react";
import { fetchForecast } from "../api/forecast";
import ForecastChart from "./ForecastChart";

const MIN_YEARS = 1;
const MAX_YEARS = 15;
const DEFAULT_YEARS = 5;
const DEBOUNCE_MS = 300;

export default function ForecastModal({ country, onClose }) {
  const [yearsAhead, setYearsAhead] = useState(DEFAULT_YEARS);
  const [forecast, setForecast] = useState(null);
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
    // Note `forecast` is deliberately NOT cleared here: the chart keeps
    // showing the last successful result (dimmed via status==="loading")
    // instead of disappearing, so the modal never resizes or blanks out
    // between requests.
    setStatus("loading");
    const timeoutId = setTimeout(() => {
      fetchForecast(country.sipriName, yearsAhead)
        .then((data) => {
          if (cancelled) return;
          setForecast(data.forecast);
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

        <div className="modal__results">
          <ForecastChart
            history={country.history}
            forecast={forecast}
            latestYear={country.latestYear}
            isLoading={status === "loading"}
          />
          <div className="forecast-chart__legend">
            <span className="forecast-chart__legend-item">
              <span className="forecast-chart__swatch forecast-chart__swatch--observed" /> Observed
            </span>
            <span className="forecast-chart__legend-item">
              <span className="forecast-chart__swatch forecast-chart__swatch--forecast" /> Forecast
            </span>
          </div>
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
