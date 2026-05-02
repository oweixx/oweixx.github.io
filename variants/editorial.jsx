// ─────────────────────────────────────────────────────────────
// Variant 2 — Editorial
// Newspaper / journal feel. Big serif display, drop-cap intro,
// rule-divided sections, two-column body for publications.
// ─────────────────────────────────────────────────────────────

const ED = window.CV_DATA;

const EditorialStyles = {
  page: {
    maxWidth: 920,
    margin: "0 auto",
    padding: "var(--s-7) var(--s-5) var(--s-9)",
    background: "var(--paper)",
  },

  // Masthead — newspaper top bar
  masthead: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
    paddingBottom: "var(--s-2)",
    borderBottom: "3px double var(--ink)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--t-xs)",
    color: "var(--ink-3)",
    textTransform: "uppercase",
    letterSpacing: "0.1em",
    marginBottom: "var(--s-6)",
  },

  // Hero — display name, kicker, lede
  hero: { paddingBottom: "var(--s-6)", borderBottom: "1px solid var(--rule)" },
  kicker: {
    fontFamily: "var(--font-mono)",
    fontSize: "var(--t-xs)",
    color: "var(--accent)",
    textTransform: "uppercase",
    letterSpacing: "0.14em",
    marginBottom: "var(--s-3)",
  },
  display: {
    fontFamily: "var(--font-serif)",
    fontSize: "var(--t-display)",
    fontWeight: 400,
    lineHeight: 0.96,
    letterSpacing: "-0.025em",
    color: "var(--ink)",
    fontStyle: "italic",
  },
  lede: {
    display: "grid",
    gridTemplateColumns: "1.4fr 1fr",
    gap: "var(--s-7)",
    marginTop: "var(--s-5)",
    alignItems: "start",
  },
  ledeBody: {
    fontFamily: "var(--font-serif)",
    fontSize: "var(--t-md)",
    lineHeight: 1.55,
    color: "var(--ink-2)",
    columnCount: 1,
  },
  ledeBodyP: { marginBottom: "var(--s-3)" },
  dropcap: {
    float: "left",
    fontFamily: "var(--font-serif)",
    fontSize: 76,
    lineHeight: 0.85,
    fontWeight: 500,
    color: "var(--accent)",
    paddingRight: 8,
    paddingTop: 6,
    fontStyle: "normal",
  },
  ledeMeta: {
    borderLeft: "1px solid var(--rule)",
    paddingLeft: "var(--s-5)",
    fontSize: "var(--t-sm)",
  },
  ledeMetaRow: { display: "grid", gridTemplateColumns: "70px 1fr", gap: "var(--s-3)", padding: "var(--s-2) 0", borderBottom: "1px dotted var(--rule)" },
  ledeMetaLabel: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em" },
  ledeMetaVal: { color: "var(--ink-2)" },

  // Section — rule-heavy, ALL CAPS small kicker + serif title
  sec: { marginTop: "var(--s-8)" },
  secHead: {
    display: "flex",
    alignItems: "baseline",
    gap: "var(--s-4)",
    paddingBottom: "var(--s-3)",
    borderBottom: "1px solid var(--ink)",
    marginBottom: "var(--s-5)",
  },
  secNum: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--accent)", letterSpacing: "0.1em" },
  secTitle: { fontFamily: "var(--font-serif)", fontSize: "var(--t-2xl)", fontWeight: 400, fontStyle: "italic", letterSpacing: "-0.01em" },
  secAside: { marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.1em" },

  // Experience — editorial: row with org as headline, role italic
  expGrid: { display: "grid", gridTemplateColumns: "1fr", gap: 0 },
  expRow: { display: "grid", gridTemplateColumns: "120px 1fr 200px", gap: "var(--s-4)", padding: "var(--s-4) 0", borderBottom: "1px solid var(--rule-2)" },
  expPeriod: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em", paddingTop: 4 },
  expOrg: { fontFamily: "var(--font-serif)", fontSize: "var(--t-md)", fontWeight: 500, color: "var(--ink)" },
  expRole: { fontFamily: "var(--font-serif)", fontStyle: "italic", color: "var(--ink-2)", fontSize: "var(--t-sm)", marginTop: 2 },
  expNote: { color: "var(--ink-3)", fontSize: "var(--t-sm)", marginTop: 4 },
  expLoc: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", paddingTop: 4, textAlign: "right" },

  // Highlights — magazine grid, big imagery
  hGrid: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "var(--s-5)" },
  hCard: { display: "block" },
  hThumb: { aspectRatio: "16 / 10", overflow: "hidden", background: "var(--paper-2)" },
  hTitle: { fontFamily: "var(--font-serif)", fontSize: "var(--t-xl)", marginTop: "var(--s-3)", color: "var(--ink)", lineHeight: 1.2 },
  hVenue: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--accent)", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.1em" },

  // Publications — two-column with year column
  pubGrid: { display: "grid", gridTemplateColumns: "1fr", gap: 0, columnGap: "var(--s-7)" },
  pubRow: { display: "grid", gridTemplateColumns: "60px 1fr", gap: "var(--s-4)", padding: "var(--s-4) 0", borderBottom: "1px solid var(--rule-2)", breakInside: "avoid" },
  pubYear: { fontFamily: "var(--font-serif)", fontSize: "var(--t-xl)", fontStyle: "italic", color: "var(--accent)", lineHeight: 1, paddingTop: 2 },
  pubTitle: { fontFamily: "var(--font-serif)", fontSize: "var(--t-md)", color: "var(--ink)", fontWeight: 500, lineHeight: 1.3 },
  pubAuthors: { fontSize: "var(--t-sm)", color: "var(--ink-2)", marginTop: 4 },
  pubVenue: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.06em" },
  pubLinks: { display: "flex", gap: "var(--s-2)", marginTop: "var(--s-2)" },
  pubLinkA: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink)", textTransform: "uppercase", letterSpacing: "0.06em", padding: "2px 8px", border: "1px solid var(--rule)", borderRadius: 999 },

  // Awards — inline list with em-dashes
  awardsBlock: { fontFamily: "var(--font-serif)", fontSize: "var(--t-md)", lineHeight: 1.7, color: "var(--ink-2)", columnCount: 2, columnGap: "var(--s-7)" },
  awardLi: { breakInside: "avoid", paddingBottom: "var(--s-2)" },
  awardOrn: { color: "var(--accent)", fontStyle: "italic" },

  // Services
  servGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--s-7)" },
  servCol: {},
  servLabel: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--s-2)" },
  servList: { fontSize: "var(--t-sm)", color: "var(--ink-2)", lineHeight: 1.8 },

  colophon: { marginTop: "var(--s-8)", paddingTop: "var(--s-4)", borderTop: "3px double var(--ink)", fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.1em", display: "flex", justifyContent: "space-between" },
};

function EdPlaceholder({ label, style }) {
  return <div className="cv-placeholder" style={style}>{label}</div>;
}

function EditorialCV() {
  const es = EditorialStyles;
  const today = new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  // Split lede paragraphs to drop-cap the first one
  const [first, ...rest] = ED.identity.bio;

  return (
    <div className="cv-reset" style={es.page}>
      <div style={es.masthead}>
        <span>The Curriculum Vitæ</span>
        <span>Vol. I · No. 1</span>
        <span>{today}</span>
      </div>

      <header style={es.hero}>
        <div style={es.kicker}>{ED.identity.title}</div>
        <h1 style={es.display}>{ED.identity.name}</h1>
        <div style={es.lede}>
          <div style={es.ledeBody}>
            <p style={es.ledeBodyP}>
              <span style={es.dropcap}>{(first || "I").trim().charAt(0)}</span>
              {(first || "").slice(1)}
            </p>
            {rest.map((p, i) => <p key={i} style={es.ledeBodyP}>{p}</p>)}
          </div>
          <div style={es.ledeMeta}>
            <div style={es.ledeMetaRow}>
              <span style={es.ledeMetaLabel}>Affil.</span>
              <span style={es.ledeMetaVal}>{ED.identity.affiliation}</span>
            </div>
            <div style={es.ledeMetaRow}>
              <span style={es.ledeMetaLabel}>Email</span>
              <span style={es.ledeMetaVal}>{ED.identity.email}</span>
            </div>
            <div style={es.ledeMetaRow}>
              <span style={es.ledeMetaLabel}>Find</span>
              <span style={es.ledeMetaVal}>
                {ED.links.map((l, i) => (
                  <React.Fragment key={i}>
                    {i > 0 && " · "}
                    <a href={l.href} style={{ borderBottom: "1px solid var(--rule)" }}>{l.label}</a>
                  </React.Fragment>
                ))}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Experience */}
      <section style={es.sec}>
        <div style={es.secHead}>
          <span style={es.secNum}>I.</span>
          <h2 style={es.secTitle}>Experience</h2>
          <span style={es.secAside}>{ED.experience.length} positions</span>
        </div>
        <div style={es.expGrid}>
          {ED.experience.map((e, i) => (
            <div key={i} style={es.expRow}>
              <div style={es.expPeriod}>{e.period}</div>
              <div>
                <div style={es.expOrg}>{e.org}</div>
                <div style={es.expRole}>{e.role}</div>
                <div style={es.expNote}>{e.note}</div>
              </div>
              <div style={es.expLoc}>{e.location}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Highlights */}
      <section style={es.sec}>
        <div style={es.secHead}>
          <span style={es.secNum}>II.</span>
          <h2 style={es.secTitle}>Highlighted Research</h2>
        </div>
        <div style={es.hGrid}>
          {ED.highlights.map((h, i) => (
            <a key={i} href={h.href} style={es.hCard}>
              <div style={es.hThumb}>
                {h.thumb
                  ? <img src={h.thumb} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                  : <EdPlaceholder label="figure" style={{ width: "100%", height: "100%" }} />}
              </div>
              <div style={es.hTitle}>{h.title}</div>
              <div style={es.hVenue}>{h.venue}</div>
            </a>
          ))}
        </div>
      </section>

      {/* Publications */}
      <section style={es.sec}>
        <div style={es.secHead}>
          <span style={es.secNum}>III.</span>
          <h2 style={es.secTitle}>Selected Publications</h2>
        </div>
        <div style={es.pubGrid}>
          {ED.publications.map((p, i) => (
            <div key={i} style={es.pubRow}>
              <div style={es.pubYear}>’{p.year.slice(-2)}</div>
              <div>
                <div style={es.pubTitle}>{p.title}</div>
                <div style={es.pubAuthors}>{p.authors}</div>
                <div style={es.pubVenue}>{p.venue}</div>
                <div style={es.pubLinks}>
                  {p.links.map((l, j) => <a key={j} href={l.href} style={es.pubLinkA}>{l.label}</a>)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Awards */}
      <section style={es.sec}>
        <div style={es.secHead}>
          <span style={es.secNum}>IV.</span>
          <h2 style={es.secTitle}>Awards & Honors</h2>
        </div>
        <ul style={es.awardsBlock}>
          {ED.awards.map((a, i) => (
            <li key={i} style={es.awardLi}>
              <span style={es.awardOrn}>§ </span>{a}
            </li>
          ))}
        </ul>
      </section>

      {/* Services */}
      <section style={es.sec}>
        <div style={es.secHead}>
          <span style={es.secNum}>V.</span>
          <h2 style={es.secTitle}>Academic Services</h2>
        </div>
        <div style={es.servGrid}>
          <div style={es.servCol}>
            <div style={es.servLabel}>Conferences</div>
            <div style={es.servList}>
              {ED.services.conferences.map((s, i) => <div key={i}>{s}</div>)}
            </div>
          </div>
          <div style={es.servCol}>
            <div style={es.servLabel}>Journals</div>
            <div style={es.servList}>
              {ED.services.journals.map((s, i) => <div key={i}>{s}</div>)}
            </div>
          </div>
        </div>
      </section>

      <div style={es.colophon}>
        <span>Set in Source Serif & Inter Tight</span>
        <span>—</span>
        <span>{ED.identity.name}</span>
      </div>
    </div>
  );
}

window.EditorialCV = EditorialCV;
