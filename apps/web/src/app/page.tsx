import Link from "next/link";
import { Nav } from "@/components/nav";

export default function HomePage() {
  return (
    <main className="shell">
      <Nav />
      <section className="grid-asym" style={{ alignItems: "end", minHeight: "72vh" }}>
        <div style={{ gridColumn: "1 / span 9" }}>
          <p className="section-kicker">A live rating-aware cinema engine</p>
          <h1 className="display">
            Rate what
            <br />
            moved you.
          </h1>
        </div>
        <div style={{ gridColumn: "9 / span 4", marginBottom: 20 }}>
          <div className="accent-tick" />
          <p style={{ fontSize: 18, lineHeight: 1.35, maxWidth: 360 }}>
            TenFlix turns your ratings into a living taste profile, then hands you films
            with reasons and places to watch.
          </p>
          <Link className="voltage-button" href="/app">
            Build your taste profile →
          </Link>
        </div>
      </section>
      <section style={{ marginTop: 120 }}>
        <p className="section-kicker">The interaction</p>
        <h2
          style={{
            fontFamily: "var(--font-pp-mondwest)",
            fontSize: "clamp(56px, 10vw, 160px)",
            lineHeight: 0.9,
            letterSpacing: "-0.04em",
            margin: 0
          }}
        >
          Drag films into feeling.
        </h2>
      </section>
    </main>
  );
}
