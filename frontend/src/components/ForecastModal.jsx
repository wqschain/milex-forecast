import { useEffect, useState } from "react";
import { fetchForecast } from "../api/forecast";

const MIN_YEARS = 1;
const MAX_YEARS = 15;
const DEFAULT_YEARS = 5;

export default function ForecastModal({ country, onClose }) {
  const [yearsAhead, setYearsAhead] = useState(DEFAULT_YEARS);
  const [forecast, setForecast] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | error | done
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let cancelled = false;

    setStatus("loading");
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

    return () => {
      cancelled = true;
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
          Most recent known spending: {(country.latestSpending * 100).toFixed(2)}% of GDP ({country.latestYear})
        </p>

        <label className="modal__slider-label" htmlFor="years-ahead">
          Forecast {yearsAhead} year{yearsAhead === 1 ? "" : "s"} ahead
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
          {status === "loading" && <p>Loading forecast...</p>}
          {status === "error" && <p className="modal__error">Failed to load forecast: {errorMessage}</p>}
          {status === "done" && forecast && (
            <table className="modal__table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Forecasted spending (% of GDP)</th>
                </tr>
              </thead>
              <tbody>
                {forecast.map((value, i) => (
                  <tr key={country.latestYear + 1 + i}>
                    <td>{country.latestYear + 1 + i}</td>
                    <td>{(value * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
