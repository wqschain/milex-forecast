import { useEffect } from "react";
import ForecastCard from "./ForecastCard";

/**
 * The always-visible edge tab plus the full-screen overlay it opens.
 * Rendered unconditionally (not gated behind `isOpen` itself) so the
 * overlay's contents - and every pinned ForecastCard's fetch - stay
 * mounted and running in the background while closed; only its position is
 * toggled (translated off-screen vs. into view). That's what makes Ukraine
 * "pre-loaded and ready to view immediately" on first open true in
 * practice, not just in wording - its fetch started the moment the app
 * loaded (see App.jsx's DEFAULT_PINNED), regardless of whether anyone has
 * opened the panel yet.
 */
export default function ComparisonPanel({ pinnedCountries, onRemove, isOpen, onToggle, onClose }) {
  useEffect(() => {
    if (!isOpen) return;
    function handleKeyDown(evt) {
      if (evt.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      <button className="comparison-tab" onClick={onToggle} aria-expanded={isOpen}>
        Comparison{pinnedCountries.length > 0 ? ` (${pinnedCountries.length})` : ""}
      </button>

      <div className={`comparison-overlay${isOpen ? " comparison-overlay--open" : ""}`} aria-hidden={!isOpen}>
        <div className="comparison-overlay__header">
          <div>
            <h2>Comparison</h2>
            <p>Click a highlighted country on the map to add it here.</p>
          </div>
          <button className="comparison-overlay__close" onClick={onClose} aria-label="Close comparison panel">
            &times;
          </button>
        </div>
        {pinnedCountries.length === 0 ? (
          <p className="comparison-panel__empty">No countries pinned yet.</p>
        ) : (
          <div className="comparison-grid">
            {pinnedCountries.map((country) => (
              <ForecastCard key={country.sipriName} country={country} onRemove={() => onRemove(country.sipriName)} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}
