import { useRef, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import worldTopology from "world-atlas/countries-110m.json";
import { countryDataByMapName } from "../data/countryData";
import CountryTooltip from "./CountryTooltip";

const HIGHLIGHT_FILL = "#3f7cac";
const HIGHLIGHT_HOVER_FILL = "#2d5d82";
const NO_DATA_FILL = "#3a3a3a";

const GEOGRAPHY_STYLE = {
  default: { stroke: "#1e1e1e", strokeWidth: 0.5, outline: "none" },
  hover: { stroke: "#1e1e1e", strokeWidth: 0.5, outline: "none" },
  pressed: { stroke: "#1e1e1e", strokeWidth: 0.5, outline: "none" },
};

function geographyStyle(hasData) {
  const fill = hasData ? HIGHLIGHT_FILL : NO_DATA_FILL;
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
  const containerRef = useRef(null);

  function pointerPosition(evt) {
    const bounds = containerRef.current.getBoundingClientRect();
    return { x: evt.clientX - bounds.left, y: evt.clientY - bounds.top };
  }

  return (
    <div className="world-map" ref={containerRef}>
      <ComposableMap projectionConfig={{ scale: 147 }}>
        <Geographies geography={worldTopology}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const data = countryDataByMapName.get(geo.properties.name);
              const hasData = Boolean(data);

              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
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
                    onSelectCountry(data);
                  }}
                  style={geographyStyle(hasData)}
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
