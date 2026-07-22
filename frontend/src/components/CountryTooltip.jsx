export default function CountryTooltip({ x, y, countryName, latestYear, latestSpending }) {
  return (
    <div className="country-tooltip" style={{ left: x + 14, top: y + 14 }}>
      <div className="country-tooltip__name">{countryName}</div>
      <div className="country-tooltip__stat">
        {(latestSpending * 100).toFixed(2)}% of GDP <span className="country-tooltip__year">({latestYear})</span>
      </div>
    </div>
  );
}
