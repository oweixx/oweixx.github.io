// ─────────────────────────────────────────────────────────────
// Variant 1 — Classic
// Faithful to the source: minimal, single column, centered.
// Slight refinements: type pairing, vertical rhythm, hairlines.
// ─────────────────────────────────────────────────────────────

const D = window.CV_DATA;

const ClassicStyles = {
  page: {
    maxWidth: 720,
    margin: "0 auto",
    padding: "var(--s-8) var(--s-5) var(--s-9)",
    background: "var(--paper)",
  },
  hero: { display: "grid", gridTemplateColumns: "140px 1fr", gap: "var(--s-5)", alignItems: "start", marginBottom: "var(--s-7)" },
  photo: { width: 140, height: 140, borderRadius: "50%", overflow: "hidden", background: "var(--paper-2)" },
  name: { fontFamily: "var(--font-serif)", fontSize: "var(--t-2xl)", fontWeight: 500, lineHeight: 1.1, letterSpacing: "-0.01em", color: "var(--ink)" },
  email: { fontFamily: "var(--font-mono)", fontSize: "var(--t-sm)", color: "var(--ink-3)", marginTop: "var(--s-2)" },
  bio: { marginTop: "var(--s-4)", color: "var(--ink-2)", lineHeight: 1.62 },
  bioP: { marginBottom: "var(--s-2)" },
  links: { display: "flex", flexWrap: "wrap", gap: "0 var(--s-3)", marginTop: "var(--s-4)", fontSize: "var(--t-sm)", color: "var(--ink-2)" },
  linkSep: { color: "var(--rule)" },
  linkA: { borderBottom: "1px solid var(--rule)", paddingBottom: 1 },

  sectionTitle: {
    fontFamily: "var(--font-serif)",
    fontSize: "var(--t-lg)",
    fontWeight: 500,
    color: "var(--ink)",
    marginTop: "var(--s-7)",
    marginBottom: "var(--s-4)",
    paddingBottom: "var(--s-2)",
    borderBottom: "1px solid var(--rule)",
    letterSpacing: "-0.005em",
  },

  expRow: { display: "grid", gridTemplateColumns: "44px 1fr auto", gap: "var(--s-4)", padding: "var(--s-3) 0", borderBottom: "1px solid var(--rule-2)", alignItems: "start" },
  expLogo: { width: 44, height: 44, borderRadius: "var(--r-sm)", overflow: "hidden", background: "var(--paper-2)" },
  expRole: { fontWeight: 500, color: "var(--ink)" },
  expOrg: { color: "var(--ink-2)", fontSize: "var(--t-sm)" },
  expNote: { color: "var(--ink-3)", fontSize: "var(--t-sm)", marginTop: 2 },
  expMeta: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", whiteSpace: "nowrap", textAlign: "right" },

  highlights: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "var(--s-3)" },
  hCard: { display: "block" },
  hThumb: { aspectRatio: "1 / 1", borderRadius: "var(--r-md)", overflow: "hidden", background: "var(--paper-2)" },
  hTitle: { fontSize: "var(--t-sm)", marginTop: "var(--s-2)", color: "var(--ink)", fontWeight: 500 },
  hVenue: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)" },

  pubRow: { display: "grid", gridTemplateColumns: "100px 1fr", gap: "var(--s-4)", padding: "var(--s-4) 0", borderBottom: "1px solid var(--rule-2)" },
  pubThumb: { width: 100, height: 70, borderRadius: "var(--r-sm)", overflow: "hidden", background: "var(--paper-2)" },
  pubTitle: { fontFamily: "var(--font-serif)", fontSize: "var(--t-md)", color: "var(--ink)", fontWeight: 500, lineHeight: 1.3 },
  pubAuthors: { fontSize: "var(--t-sm)", color: "var(--ink-2)", marginTop: 4 },
  pubVenue: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.04em" },
  pubLinks: { display: "flex", gap: "var(--s-3)", marginTop: "var(--s-2)", fontSize: "var(--t-xs)", fontFamily: "var(--font-mono)" },
  pubLinkA: { color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.06em" },

  awardLi: { padding: "var(--s-2) 0", borderBottom: "1px dashed var(--rule-2)", color: "var(--ink-2)", fontSize: "var(--t-sm)", display: "flex", gap: "var(--s-3)" },
  awardDot: { color: "var(--accent)", fontFamily: "var(--font-mono)" },

  servGrid: { display: "grid", gridTemplateColumns: "120px 1fr", gap: "var(--s-3) var(--s-4)" },
  servLabel: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em", paddingTop: 2 },
  servList: { fontSize: "var(--t-sm)", color: "var(--ink-2)", lineHeight: 1.7 },
};

function Placeholder({ label, style }) {
  return <div className="cv-placeholder" style={style}>{label}</div>;
}

function ClassicCV() {
  const cs = ClassicStyles;
  return (
    <div className="cv-reset" style={cs.page}>
      {/* Hero */}
      <header style={cs.hero}>
        <div style={cs.photo}>
          {D.identity.photo
            ? <img src={D.identity.photo} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            : <Placeholder label="photo" style={{ width: "100%", height: "100%" }} />}
        </div>
        <div>
          <h1 style={cs.name}>{D.identity.name}</h1>
          <div style={cs.email}>{D.identity.email}</div>
          <div style={cs.bio}>
            {D.identity.bio.map((p, i) => <p key={i} style={cs.bioP}>{p}</p>)}
          </div>
          <div style={cs.links}>
            {D.links.map((l, i) => (
              <React.Fragment key={i}>
                {i > 0 && <span style={cs.linkSep}>/</span>}
                <a href={l.href} style={cs.linkA}>{l.label}</a>
              </React.Fragment>
            ))}
          </div>
        </div>
      </header>

      {/* Experience */}
      <h2 style={cs.sectionTitle}>Experience</h2>
      <div>
        {D.experience.map((e, i) => (
          <div key={i} style={cs.expRow}>
            <div style={cs.expLogo}>
              {e.logo
                ? <img src={e.logo} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                : <Placeholder label="logo" style={{ width: "100%", height: "100%", fontSize: 9 }} />}
            </div>
            <div>
              <div style={cs.expRole}>{e.role} <span style={{ color: "var(--ink-3)", fontWeight: 400 }}>· {e.org}</span></div>
              <div style={cs.expOrg}>{e.location}</div>
              <div style={cs.expNote}>{e.note}</div>
            </div>
            <div style={cs.expMeta}>{e.period}</div>
          </div>
        ))}
      </div>

      {/* Highlighted Research */}
      <h2 style={cs.sectionTitle}>Highlighted Research</h2>
      <div style={cs.highlights}>
        {D.highlights.map((h, i) => (
          <a key={i} style={cs.hCard} href={h.href}>
            <div style={cs.hThumb}>
              {h.thumb
                ? <img src={h.thumb} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                : <Placeholder label="thumb" style={{ width: "100%", height: "100%" }} />}
            </div>
            <div style={cs.hTitle}>{h.title}</div>
            <div style={cs.hVenue}>{h.venue}</div>
          </a>
        ))}
      </div>

      {/* Publications */}
      <h2 style={cs.sectionTitle}>Publications</h2>
      <div>
        {D.publications.map((p, i) => (
          <div key={i} style={cs.pubRow}>
            <div style={cs.pubThumb}>
              {p.thumb
                ? <img src={p.thumb} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                : <Placeholder label="figure" style={{ width: "100%", height: "100%", fontSize: 9 }} />}
            </div>
            <div>
              <div style={cs.pubTitle}>{p.title}</div>
              <div style={cs.pubAuthors}>{p.authors}</div>
              <div style={cs.pubVenue}>{p.venue} · {p.year}</div>
              <div style={cs.pubLinks}>
                {p.links.map((l, j) => <a key={j} href={l.href} style={cs.pubLinkA}>{l.label}</a>)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Awards */}
      <h2 style={cs.sectionTitle}>Awards & Honors</h2>
      <ul>
        {D.awards.map((a, i) => (
          <li key={i} style={cs.awardLi}>
            <span style={cs.awardDot}>{String(i + 1).padStart(2, "0")}</span>
            <span>{a}</span>
          </li>
        ))}
      </ul>

      {/* Academic Services */}
      <h2 style={cs.sectionTitle}>Academic Services</h2>
      <div style={cs.servGrid}>
        <div style={cs.servLabel}>Conferences</div>
        <div style={cs.servList}>
          {D.services.conferences.map((s, i) => <div key={i}>{s}</div>)}
        </div>
        <div style={cs.servLabel}>Journals</div>
        <div style={cs.servList}>
          {D.services.journals.map((s, i) => <div key={i}>{s}</div>)}
        </div>
      </div>

      {/* News */}
      <h2 style={cs.sectionTitle}>News</h2>
      <div>
        {D.news.map((n, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "100px 1fr", gap: "var(--s-4)", padding: "var(--s-2) 0", borderBottom: "1px dashed var(--rule-2)", fontSize: "var(--t-sm)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)" }}>{n.date}</div>
            <div style={{ color: "var(--ink-2)" }}>{n.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

window.ClassicCV = ClassicCV;
