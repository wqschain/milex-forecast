/**
 * `onAdd` is only passed on touch devices, mid-preview (see WorldMap.jsx's
 * tap-to-preview flow) - its presence is what switches the tooltip from a
 * pure, non-interactive hover readout (pointer-events: none, the mouse
 * behavior, unchanged) to a tappable card with an explicit action, since a
 * touch user has no hover state to fall back on for "how do I add this."
 */
export default function CountryTooltip({ x, y, countryName, latestYear, latestSpending, onAdd }) {
  return (
    <div
      className={`country-tooltip${onAdd ? " country-tooltip--interactive" : ""}`}
      style={{ left: x + 14, top: y + 14 }}
    >
      <div className="country-tooltip__name">{countryName}</div>
      <div className="country-tooltip__stat">
        <span className="mono country-tooltip__value">{(latestSpending * 100).toFixed(2)}%</span> of GDP{" "}
        <span className="mono country-tooltip__year">({latestYear})</span>
      </div>
      {onAdd && (
        <button type="button" className="country-tooltip__add" onClick={onAdd}>
          + Add to comparison
        </button>
      )}
    </div>
  );
}
