import { useEffect } from "react";

/**
 * Mirrors ComparisonPanel's tab-and-overlay pattern exactly (same trigger
 * style, same open/close behavior via the tab/×/Escape) but on the
 * opposite edge of the screen and with static content instead of pinned
 * countries - there's no per-item state here, just a fixed writeup.
 */
export default function InfoPanel({ isOpen, onToggle, onClose }) {
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
      <button className="info-tab" onClick={onToggle} aria-expanded={isOpen}>
        Info
      </button>

      <div className={`info-overlay${isOpen ? " info-overlay--open" : ""}`} aria-hidden={!isOpen}>
        <div className="info-overlay__header">
          <h2>About this tool</h2>
          <button className="info-overlay__close" onClick={onClose} aria-label="Close info panel">
            &times;
          </button>
        </div>

        <div className="info-overlay__body">
          <section>
            <h3>Data</h3>
            <p>
              31 countries, 1980&ndash;2025, from SIPRI&rsquo;s military expenditure database. Spending is measured
              as a share of GDP. A handful of countries were left out for lacking a reliable historical record
              (North Korea, for one) &ndash; this isn&rsquo;t every country&rsquo;s military, just the ones with
              real data to forecast from.
            </p>
          </section>

          <section>
            <h3>Forecasting</h3>
            <p>
              Two models are tested for every country: a simple linear trend, and Prophet, which can detect real
              shifts in a country&rsquo;s trajectory. Both are backtested on held-out data (trained through 2015,
              checked against the real 2016&ndash;2025 values), and whichever measures lower error is used &ndash;
              not one model applied uniformly. Today that&rsquo;s Prophet for 30 of 31 countries; linear regression
              still wins for T&uuml;rkiye, on its own measured accuracy.
            </p>
          </section>

          <section>
            <h3>Scenarios</h3>
            <p>
              A few countries have spending histories dominated by one exceptional event &ndash; Ukraine&rsquo;s
              2022 invasion, T&uuml;rkiye&rsquo;s currency crisis since 2018 &ndash; where a single extrapolated
              trend line would be dishonest (Ukraine&rsquo;s alone would cross 100% of GDP by the 2030s). These
              countries get named scenarios instead: two or three explicitly labeled forecasts, each conditional on
              a stated assumption (&ldquo;conflict continues,&rdquo; &ldquo;conflict resolves within 3
              years&rdquo;), plus a growth cap grounded in real historical precedent rather than an unconstrained
              trend.
            </p>
          </section>

          <section>
            <h3>What&rsquo;s actually proven</h3>
            <p>
              An automated pipeline exists to extend scenario treatment to future flagged countries without
              hand-research each time. Its detection and categorization stages were validated with a genuinely
              blind test against Ukraine &ndash; calibrated on the other 30 countries, given no hardcoded dates or
              country name, checked against the known answer only after every stage had already run &ndash; and it
              correctly identified the real conflict episode and category. The same pipeline was also run against
              Nigeria and Australia to confirm it executes end-to-end, but neither is an actual flagged country in
              production; there&rsquo;s no ground truth to check those results against, and none is claimed. Only
              Ukraine and T&uuml;rkiye have real scenario forecasting today.
            </p>
          </section>
        </div>
      </div>
    </>
  );
}
