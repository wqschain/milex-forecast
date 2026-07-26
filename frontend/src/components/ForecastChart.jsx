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

// One style per forecast line, in order - all in the accent (burgundy)
// family, distinguished by lightness AND dash pattern (not color alone),
// so multiple scenario lines stay visually a set of "forecasts" rather than
// introducing unrelated hues. A single-line trend/undetermined forecast
// always gets SCENARIO_STYLES[0], which exactly matches this chart's
// original single-forecast look - no visual change for the ~29 countries
// that only ever show one line.
export const SCENARIO_STYLES = [
  { color: "#7d3644", dash: "5 3" }, // --accent
  { color: "#a86773", dash: "2 2" }, // --accent-dim
  { color: "#5c2733", dash: "1 3" }, // --accent-deep
];

function buildChartData(history, series, latestYear) {
  const recentHistory = history.slice(-HISTORY_WINDOW_YEARS);
  const points = recentHistory.map((h) => {
    const point = { year: h.year, observed: h.spending };
    series.forEach((s) => {
      point[s.key] = null;
    });
    return point;
  });

  if (points.length > 0) {
    // Bridge point: carries both the observed value and every series' value
    // at the same year, so the ink (observed) segment connects to each
    // burgundy (forecast) segment with no gap.
    const last = points[points.length - 1];
    series.forEach((s) => {
      last[s.key] = last.observed;
    });
  }

  const horizon = series.reduce((max, s) => Math.max(max, s.values.length), 0);
  for (let i = 0; i < horizon; i++) {
    const point = { year: latestYear + 1 + i, observed: null };
    series.forEach((s) => {
      point[s.key] = s.values[i] ?? null;
    });
    points.push(point);
  }

  return points;
}

function formatPercent(value) {
  return value == null ? "–" : `${(value * 100).toFixed(2)}%`;
}

/**
 * `series`: array of { key, label, values } - one entry renders exactly
 * like the original single-forecast chart; two or three render as multiple
 * named lines (scenario status), sharing one observed history line.
 */
export default function ForecastChart({ history, series, latestYear, isLoading }) {
  const data = buildChartData(history, series, latestYear);

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
          formatter={(value, name) => [formatPercent(value), name === "observed" ? "Observed" : name]}
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
        {series.map((s, i) => {
          const style = SCENARIO_STYLES[i % SCENARIO_STYLES.length];
          return (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={style.color}
              strokeWidth={2}
              strokeDasharray={style.dash}
              dot={false}
              connectNulls={false}
              animationDuration={300}
            />
          );
        })}
      </LineChart>
    </div>
  );
}
