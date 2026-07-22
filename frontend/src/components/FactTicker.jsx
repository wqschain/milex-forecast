import { useEffect, useState } from "react";
import facts from "../data/facts.json";

const CYCLE_MS = 5000;

export default function FactTicker() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setIndex((i) => (i + 1) % facts.length);
    }, CYCLE_MS);
    return () => clearInterval(id);
  }, []);

  if (facts.length === 0) return null;

  return (
    <div className="fact-ticker">
      <span className="fact-ticker__label">FIELD NOTES</span>
      <span className="fact-ticker__divider" />
      <span key={index} className="fact-ticker__text mono">
        {facts[index]}
      </span>
    </div>
  );
}
