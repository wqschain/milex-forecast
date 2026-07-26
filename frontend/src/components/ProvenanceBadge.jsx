function humanize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Surfaces a source/confidence pair (scenario_source+scenario_confidence,
 * or cap_source+cap_confidence) visibly, not just in the API response.
 * "automated" gets a distinct, noticeable treatment - never rendered with
 * the same visual weight as a hand-researched/hand-built value, since an
 * unverified GDELT-only estimate and an externally-checked one must be
 * tellable apart at a glance, not just in the data.
 */
export default function ProvenanceBadge({ label, source, confidence }) {
  const isAutomated = source === "automated";
  return (
    <span className={`provenance-badge${isAutomated ? " provenance-badge--automated" : ""}`}>
      <span className="provenance-badge__label">{label}</span>
      <span className="provenance-badge__value">
        {humanize(source)} &middot; {humanize(confidence)} confidence
      </span>
    </span>
  );
}
