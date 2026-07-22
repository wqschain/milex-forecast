// Maps SIPRI country names (as used in backend/model_selection_log.json and
// the /forecast API) to the "name" property used by react-simple-maps' map
// data (world-atlas, sourced from Natural Earth).
//
// Verified against both the 110m and 50m world-atlas topojson files: 27 of
// 31 SIPRI names match the map data exactly (including "United States of
// America", which happens to match despite differing conventions
// elsewhere). Only these 4 need an explicit translation:
export const SIPRI_TO_MAP_NAME = {
  "Türkiye": "Turkey",
  "Korea, South": "South Korea",
  "Viet Nam": "Vietnam",
};

export function toMapName(sipriName) {
  return SIPRI_TO_MAP_NAME[sipriName] ?? sipriName;
}
