import { useEffect, useState } from "react";

// Checked once at module load, not per keystroke or per render - this is a
// single-page app with no routing, so a module-level constant also
// guarantees the animation can never be re-triggered later in the session;
// there's nothing that re-evaluates it after the module first loads.
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Types `text` out character by character - a document generating itself,
 * not a theatrical reveal. Renders only the animated span (aria-hidden);
 * the caller wraps it in a real heading/paragraph with a full-text
 * aria-label, so screen readers get the whole string immediately rather
 * than reading it out character-by-character.
 *
 * `active` gates when typing starts (default true - starts immediately on
 * mount) - passing `active={false}` until some prior condition is met is
 * what lets two of these chain into one sequential effect (see App.jsx:
 * the subtitle's TypedText only goes active once the title's onDone
 * fires). `onDone` fires exactly once, including immediately under
 * prefers-reduced-motion (skip straight to final text, but the caller
 * still gets notified so a chained follow-up isn't left waiting forever).
 */
export default function TypedText({ text, charIntervalMs = 35, active = true, onDone }) {
  const [charCount, setCharCount] = useState(prefersReducedMotion ? text.length : 0);
  const [done, setDone] = useState(prefersReducedMotion);

  useEffect(() => {
    if (prefersReducedMotion) {
      onDone?.();
      return;
    }
    if (!active) return;
    // Side effects (onDone, which sets state on the parent) live here, as
    // plain statements in the interval callback - not inside setCharCount's
    // updater function. React expects updater functions to be pure
    // (StrictMode deliberately double-invokes them to catch exactly this
    // kind of impurity), and a setState-on-another-component call embedded
    // in one is what was triggering "Cannot update a component while
    // rendering a different component."
    let count = 0;
    const intervalId = setInterval(() => {
      count += 1;
      setCharCount(count);
      if (count >= text.length) {
        clearInterval(intervalId);
        setDone(true);
        onDone?.();
      }
    }, charIntervalMs);
    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  return (
    <span aria-hidden="true">
      {text.slice(0, charCount)}
      {!done && <span className="masthead-title__cursor" />}
    </span>
  );
}
