import { CartesianGrid, Line, LineChart, XAxis, YAxis, Tooltip } from "recharts";

// Fixed regardless of how many years are selected - more years means more
// points plotted within this same space, never a bigger chart. The modal's
// content width is exactly 420px (480px max-width minus 30px padding on
// each side); 410px leaves a small safety margin against that edge.
const CHART_WIDTH = 410;
const CHART_HEIGHT = 200;

// Trailing window of history shown alongside the forecast, so the chart
// stays readable (and the forecast portion stays visible) regardless of how
// far back a country's data goes.
const HISTORY_WINDOW_YEARS = 15;

const INK = "#1f3a5c";
const ACCENT = "#7d3644";

function buildChartData(history, forecastValues, latestYear) {
  const recentHistory = history.slice(-HISTORY_WINDOW_YEARS);
  const points = recentHistory.map((h) => ({ year: h.year, observed: h.spending, forecast: null }));

  if (points.length > 0) {
    // Bridge point: carries both series' values at the same year, so the
    // ink (observed) and burgundy (forecast) segments connect with no gap.
    points[points.length - 1] = { ...points[points.length - 1], forecast: points[points.length - 1].observed };
  }

  forecastValues.forEach((value, i) => {
    points.push({ year: latestYear + 1 + i, observed: null, forecast: value });
  });

  return points;
}

function formatPercent(value) {
  return value == null ? "–" : `${(value * 100).toFixed(2)}%`;
}

export default function ForecastChart({ history, forecast, latestYear, isLoading }) {
  const data = buildChartData(history, forecast ?? [], latestYear);

  return (
    <div className={`forecast-chart${isLoading ? " forecast-chart--loading" : ""}`}>
      <LineChart
        width={CHART_WIDTH}
        height={CHART_HEIGHT}
        data={data}
        margin={{ top: 8, right: 14, bottom: 0, left: -16 }}
      >
        <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="year"
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)", fill: "var(--text-muted)" }}
          tickLine={false}
          axisLine={{ stroke: "var(--border-strong)" }}
          minTickGap={28}
        />
        <YAxis
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fontSize: 11, fontFamily: "var(--font-mono)", fill: "var(--text-muted)" }}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        <Tooltip
          formatter={(value, name) => [formatPercent(value), name === "observed" ? "Observed" : "Forecast"]}
          labelFormatter={(year) => year}
          contentStyle={{
            fontFamily: "var(--font-ui)",
            fontSize: 12,
            border: "1px solid var(--border-strong)",
            background: "var(--bg-raised)",
            borderRadius: 4,
          }}
        />
        <Line
          type="monotone"
          dataKey="observed"
          stroke={INK}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
          animationDuration={300}
        />
        <Line
          type="monotone"
          dataKey="forecast"
          stroke={ACCENT}
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={false}
          connectNulls={false}
          animationDuration={300}
        />
      </LineChart>
    </div>
  );
}
