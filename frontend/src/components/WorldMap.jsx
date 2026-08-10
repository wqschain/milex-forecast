import { useRef, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import worldTopology from "world-atlas/countries-110m.json";
import { countryDataByMapName } from "../data/countryData";
import CountryTooltip from "./CountryTooltip";

// Ink-navy marks countries with real historical data + a trained model;
// kept deliberately muted so it reads as "measured," not as a UI accent
// color. The reserved accent color (see App.css) is used only for
// forecasted/projected values, never here.
const HIGHLIGHT_FILL = "#1f3a5c";
const HIGHLIGHT_HOVER_FILL = "#2c4f78";
const HIGHLIGHT_CLICK_FILL = "#3a6088";
const NO_DATA_FILL = "url(#no-data-hatch)";
// Dark enough to stay visible against the light no-data hatch (the majority
// of countries), while still reading as a thin separator against the ink
// navy fill of the highlighted ones.
const STROKE = "#4a4030";

const TRANSITION = "fill 250ms ease";

const GEOGRAPHY_STYLE = {
  default: { stroke: STROKE, strokeWidth: 0.5, outline: "none", transition: TRANSITION },
  hover: { stroke: STROKE, strokeWidth: 0.5, outline: "none", transition: TRANSITION },
  pressed: { stroke: STROKE, strokeWidth: 0.5, outline: "none", transition: TRANSITION },
};

function geographyStyle(hasData, isClicked) {
  const fill = hasData ? (isClicked ? HIGHLIGHT_CLICK_FILL : HIGHLIGHT_FILL) : NO_DATA_FILL;
  const hoverFill = hasData ? HIGHLIGHT_HOVER_FILL : NO_DATA_FILL;
  const cursor = hasData ? "pointer" : "default";
  return {
    default: { ...GEOGRAPHY_STYLE.default, fill, cursor },
    hover: { ...GEOGRAPHY_STYLE.hover, fill: hoverFill, cursor },
    pressed: { ...GEOGRAPHY_STYLE.pressed, fill: hoverFill, cursor },
  };
}

export default function WorldMap({ onSelectCountry }) {
  const [hovered, setHovered] = useState(null); // { data, x, y } | null
  const [pulsingKey, setPulsingKey] = useState(null);
  const containerRef = useRef(null);

  function pointerPosition(evt) {
    const bounds = containerRef.current.getBoundingClientRect();
    return { x: evt.clientX - bounds.left, y: evt.clientY - bounds.top };
  }

  return (
    <div className="world-map" ref={containerRef}>
      {/* Explicit width/height (rather than the library's 800x600/4:3
          default) so the viewBox's own aspect ratio matches how wide the
          .world-map container actually is - otherwise preserveAspectRatio
          letterboxes the map, wasting exactly the space this change is
          meant to reclaim. 800x450 (16:9) is a closer match and also a
          more natural fit for an equirectangular-ish world map than 4:3. */}
      <ComposableMap width={800} height={450} projectionConfig={{ scale: 147 }}>
        <defs>
          <pattern id="no-data-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="#e8e0cc" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="#cabf9d" strokeWidth="1.5" />
          </pattern>
        </defs>
        <Geographies geography={worldTopology}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const data = countryDataByMapName.get(geo.properties.name);
              const hasData = Boolean(data);
              const isClicked = geo.rsmKey === pulsingKey;

              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  className={isClicked ? "country-pulse" : undefined}
                  onAnimationEnd={() => setPulsingKey(null)}
                  onMouseEnter={(evt) => {
                    if (!hasData || !containerRef.current) return;
                    setHovered({ data, ...pointerPosition(evt) });
                  }}
                  onMouseMove={(evt) => {
                    if (!hasData || !containerRef.current) return;
                    setHovered({ data, ...pointerPosition(evt) });
                  }}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => {
                    if (!hasData) return;
                    setPulsingKey(geo.rsmKey);
                    onSelectCountry(data);
                  }}
                  style={geographyStyle(hasData, isClicked)}
                />
              );
            })
          }
        </Geographies>
      </ComposableMap>

      {hovered && (
        <CountryTooltip
          x={hovered.x}
          y={hovered.y}
          countryName={hovered.data.sipriName}
          latestYear={hovered.data.latestYear}
          latestSpending={hovered.data.latestSpending}
        />
      )}
    </div>
  );
}
