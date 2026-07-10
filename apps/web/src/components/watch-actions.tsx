import type { WatchAction, WatchProvider } from "@/lib/api";

export function WatchActions({
  providers = [],
  actions = []
}: {
  providers?: WatchProvider[];
  actions?: WatchAction[];
}) {
  return (
    <section style={{ marginTop: 42 }}>
      <p className="section-kicker">Available on</p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {providers.length ? (
          providers.map((provider) => (
            <a
              key={`${provider.provider_name}-${provider.provider_type}`}
              className="voltage-button"
              href={provider.deep_link ?? "#"}
              style={{ boxShadow: "none", minHeight: 44, padding: "12px 18px" }}
            >
              Watch on {provider.provider_name}
            </a>
          ))
        ) : (
          <span style={{ color: "var(--color-sage)" }}>No provider data yet.</span>
        )}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 18, marginTop: 22 }}>
        {actions.map((action) => (
          <a className="ghost-button" key={action.action_type} href={action.url}>
            {action.label} →
          </a>
        ))}
      </div>
    </section>
  );
}
