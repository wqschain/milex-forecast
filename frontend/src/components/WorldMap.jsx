import { useRef, useState } from "react";
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps";
import worldTopology from "world-atlas/countries-110m.json";
import { countryDataByMapName } from "../data/countryData";
import CountryTooltip from "./CountryTooltip";

// 1 = the initial fitted view (matches projectionConfig.scale below) - the
// floor, not an arbitrary number, so "zoom out" can never reveal blank
// space beyond the world map itself. 8 is react-simple-maps' own default
// ceiling; past that a country fills the whole viewport with nothing more
// to see, so there's no real usefulness in going further.
const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.5;
// Keeps the pannable area roughly bounded to the map's own viewBox
// (matching ComposableMap's width/height below) - without this, dragging
// at any zoom level can pull the entire map off-screen into blank space,
// the same failure mode minZoom already prevents for scroll/button zoom.
const TRANSLATE_EXTENT = [
  [0, 0],
  [800, 450],
];

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

// (hover: none) is the standard way to detect "this device has no hover
// mechanism" (touch-primary), rather than checking for touch event
// support - some hybrid devices (touchscreen laptops) support touch AND
// hover, and should keep the mouse behavior below, not the tap-to-preview
// one. Checked once at module load, not per-render - device input
// capability doesn't change at runtime.
const isTouchDevice = window.matchMedia("(hover: none)").matches;

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
  // Touch has no hover state to preview a country with before committing to
  // adding it - previewedKey tracks which country's preview is currently
  // showing (first tap), distinct from actually adding it (second tap on
  // the same country, or the tooltip's own explicit button). Unused on
  // mouse/hover devices, where onMouseEnter already does this job and a
  // single click still adds immediately, unchanged from before touch
  // support existed.
  const [previewedKey, setPreviewedKey] = useState(null);
  // Controlled center/zoom, kept in sync with user-driven gestures (wheel,
  // drag) via onMoveEnd below - not just written by the +/- buttons. That's
  // what lets a button click zoom around wherever the user last panned to,
  // instead of snapping back to the world center every time (which is what
  // happens if `center` is left at its [0,0] default while only `zoom`
  // changes).
  const [position, setPosition] = useState({ coordinates: [0, 0], zoom: 1 });
  // Neither scroll-wheel zoom nor the +/- buttons animate on their own -
  // react-simple-maps applies both as an instant transform-attribute jump
  // internally, with no exposed way to make that specific call transition.
  // Smoothing it in CSS instead (see .world-map__zoomable--smooth), gated
  // on gesture type: on for wheel, off for an actual mouse/touch drag - a
  // transition delay on drag would decouple the map's position from the
  // cursor while panning, which reads as laggy rather than smooth.
  //
  // Wheel/drag are told apart via onMove's sourceEvent (see handleMove) -
  // but the +/- buttons can't use that path at all: react-simple-maps sets
  // an internal `bypassEvents` flag before applying a prop-driven zoom
  // change specifically to suppress onMove/onMoveEnd for that call (avoids
  // a feedback loop), so onMove never fires for a button click. Confirmed
  // empirically, not assumed - a real drag correctly turned the class off,
  // but a button click afterward left it off too, since nothing was
  // telling it otherwise. Buttons set the flag explicitly themselves
  // instead of relying on an event that won't fire for them.
  const [isSmoothZoom, setIsSmoothZoom] = useState(true);
  const containerRef = useRef(null);

  function pointerPosition(evt) {
    const bounds = containerRef.current.getBoundingClientRect();
    return { x: evt.clientX - bounds.left, y: evt.clientY - bounds.top };
  }

  function handleMove(_pos, evt) {
    const sourceEvent = evt?.sourceEvent;
    setIsSmoothZoom(!sourceEvent || sourceEvent.type === "wheel");
  }

  function zoomIn() {
    setIsSmoothZoom(true);
    setPosition((pos) => ({ ...pos, zoom: Math.min(pos.zoom * ZOOM_STEP, MAX_ZOOM) }));
  }

  function addCountry(geo, data) {
    setPulsingKey(geo.rsmKey);
    setPreviewedKey(null);
    setHovered(null);
    onSelectCountry(data);
  }

  function handleGeographyClick(evt, geo, data) {
    if (!isTouchDevice) {
      addCountry(geo, data);
      return;
    }
    // Stops this from also bubbling to the map background's own onClick
    // (see below), which clears the preview on an empty-space tap - without
    // this, tapping a country would set the preview and then immediately
    // clear it again as the same event reaches the background handler.
    evt.stopPropagation();
    // First tap previews (same info a hover would show, plus an explicit
    // "add" button in the tooltip - see CountryTooltip); a second tap on
    // the same, already-previewed country adds it, mirroring "tap again to
    // confirm" without requiring the button specifically.
    if (previewedKey === geo.rsmKey) {
      addCountry(geo, data);
    } else {
      setPreviewedKey(geo.rsmKey);
      setHovered({ data, geo, ...pointerPosition(evt) });
    }
  }

  function handleBackgroundTap() {
    if (!isTouchDevice) return;
    setPreviewedKey(null);
    setHovered(null);
  }

  function zoomOut() {
    setIsSmoothZoom(true);
    setPosition((pos) => ({ ...pos, zoom: Math.max(pos.zoom / ZOOM_STEP, MIN_ZOOM) }));
  }

  return (
    <div className="world-map" ref={containerRef} onClick={handleBackgroundTap}>
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
        {/* Scroll-wheel zoom/drag-pan come from d3-zoom internally (via
            react-simple-maps' useZoomPan) the moment ZoomableGroup mounts -
            no extra wiring needed for those two. onMoveEnd is what keeps
            `position` in sync with gesture-driven moves, which is what
            makes the +/- buttons below zoom around the current pan
            position rather than fighting it. onMove (fires on every tick,
            not just at the end) drives the smooth-zoom class toggle - see
            handleMove/isSmoothZoom above. className lands on the actual
            <g transform="..."> react-simple-maps renders internally. */}
        <ZoomableGroup
          center={position.coordinates}
          zoom={position.zoom}
          minZoom={MIN_ZOOM}
          maxZoom={MAX_ZOOM}
          translateExtent={TRANSLATE_EXTENT}
          onMove={handleMove}
          onMoveEnd={setPosition}
          className={isSmoothZoom ? "world-map__zoomable--smooth" : undefined}
        >
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
                      if (isTouchDevice || !hasData || !containerRef.current) return;
                      setHovered({ data, ...pointerPosition(evt) });
                    }}
                    onMouseMove={(evt) => {
                      if (isTouchDevice || !hasData || !containerRef.current) return;
                      setHovered({ data, ...pointerPosition(evt) });
                    }}
                    onMouseLeave={() => {
                      if (isTouchDevice) return;
                      setHovered(null);
                    }}
                    onClick={(evt) => {
                      if (!hasData) return;
                      handleGeographyClick(evt, geo, data);
                    }}
                    style={geographyStyle(hasData, isClicked)}
                  />
                );
              })
            }
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>

      {hovered && (
        <CountryTooltip
          x={hovered.x}
          y={hovered.y}
          countryName={hovered.data.sipriName}
          latestYear={hovered.data.latestYear}
          latestSpending={hovered.data.latestSpending}
          onAdd={isTouchDevice && previewedKey ? () => addCountry(hovered.geo, hovered.data) : undefined}
        />
      )}

      {/* Dedicated buttons, not just scroll - a trackpad/wheel gesture
          isn't always available or comfortable, and (per the request that
          prompted this) scroll can conflict with page-level scrolling in
          contexts where the map isn't the only scrollable thing. */}
      <div className="world-map__zoom-controls">
        <button
          type="button"
          onClick={zoomIn}
          disabled={position.zoom >= MAX_ZOOM}
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={zoomOut}
          disabled={position.zoom <= MIN_ZOOM}
          aria-label="Zoom out"
        >
          &minus;
        </button>
      </div>
    </div>
  );
}
