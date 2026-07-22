export default function CountryTooltip({ x, y, countryName, latestYear, latestSpending }) {
  return (
    <div className="country-tooltip" style={{ left: x + 14, top: y + 14 }}>
      <div className="country-tooltip__name">{countryName}</div>
      <div className="country-tooltip__stat">
        <span className="mono country-tooltip__value">{(latestSpending * 100).toFixed(2)}%</span> of GDP{" "}
        <span className="mono country-tooltip__year">({latestYear})</span>
      </div>
    </div>
  );
}
