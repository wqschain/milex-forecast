import worldTopology from "world-atlas/countries-110m.json";
import countrySpending from "./countrySpending.json";
import { toMapName } from "./countryNameMap";

// Keyed by the react-simple-maps/world-atlas geography name (not the SIPRI
// name), since that's what we look up while rendering/hovering/clicking
// each Geography on the map.
export const countryDataByMapName = new Map(
  Object.entries(countrySpending).map(([sipriName, data]) => [
    toMapName(sipriName),
    { sipriName, ...data },
  ])
);

// Cross-checks our SIPRI -> map-name translation against the names actually
// present in the map's own topojson data, instead of assuming the mapping
// is correct. Runs once at module load (the topology is a static import, so
// this doesn't need to wait on any rendering) and warns about anything that
// doesn't resolve, so mismatches surface immediately during development
// rather than silently rendering a country as "no data".
function findUnmatchedCountries() {
  const mapNames = new Set(
    worldTopology.objects.countries.geometries.map((g) => g.properties.name)
  );
  return [...countryDataByMapName.entries()]
    .filter(([mapName]) => !mapNames.has(mapName))
    .map(([, data]) => data.sipriName);
}

export const unmatchedCountries = findUnmatchedCountries();

if (unmatchedCountries.length > 0) {
  console.warn(
    "[countryData] These SIPRI countries could not be confidently matched to a " +
      "map geography (check countryNameMap.js):",
    unmatchedCountries
  );
}

// Full span of the dataset (min earliest year to max latest year across all
// countries), for display in the data strip - not hardcoded, since coverage
// actually varies per country (e.g. Ukraine only starts in 1993).
const allCountryData = [...countryDataByMapName.values()];
export const datasetYearRange = {
  earliest: Math.min(...allCountryData.map((c) => c.earliestYear)),
  latest: Math.max(...allCountryData.map((c) => c.latestYear)),
};
