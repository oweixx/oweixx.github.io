// ─────────────────────────────────────────────────────────────
// Variant 3 — Compact Grid (Research Lab)
// Information-dense, two-column on wide, monospace metadata.
// Sticky side nav, tag chips, news rail, project tiles with hover.
// ─────────────────────────────────────────────────────────────

const RD = window.CV_DATA;

const LabStyles = {
  shell: {
    display: "grid",
    gridTemplateColumns: "220px 1fr",
    gap: "var(--s-7)",
    maxWidth: 1180,
    margin: "0 auto",
    padding: "var(--s-6) var(--s-5) var(--s-9)",
    background: "var(--paper)",
    minHeight: "100vh",
  },

  // Side rail
  rail: { position: "sticky", top: "var(--s-5)", alignSelf: "start", paddingTop: "var(--s-2)" },
  railPhoto: { width: 96, height: 96, borderRadius: "50%", overflow: "hidden", background: "var(--paper-2)", marginBottom: "var(--s-3)" },
  railName: { fontFamily: "var(--font-serif)", fontSize: "var(--t-lg)", fontWeight: 500, lineHeight: 1.1, letterSpacing: "-0.01em" },
  railTitle: { fontSize: "var(--t-xs)", fontFamily: "var(--font-mono)", color: "var(--ink-3)", marginTop: 4, textTransform: "uppercase", letterSpacing: "0.06em" },
  railEmail: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-2)", marginTop: "var(--s-2)", wordBreak: "break-all" },

  railNav: { marginTop: "var(--s-5)", borderTop: "1px solid var(--rule-2)", paddingTop: "var(--s-3)" },
  navItem: { display: "flex", justifyContent: "space-between", padding: "5px 0", fontSize: "var(--t-xs)", fontFamily: "var(--font-mono)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em", borderBottom: "1px dashed var(--rule-2)" },
  navItemNum: { color: "var(--accent)" },

  railLinks: { display: "flex", flexWrap: "wrap", gap: 4, marginTop: "var(--s-4)" },
  railChip: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", padding: "3px 8px", border: "1px solid var(--rule)", borderRadius: 999, color: "var(--ink-2)" },

  // Main column
  main: { minWidth: 0 },

  hero: { paddingBottom: "var(--s-5)", borderBottom: "1px solid var(--rule)", marginBottom: "var(--s-6)" },
  heroLine: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--s-3)", display: "flex", alignItems: "center", gap: "var(--s-2)" },
  heroDot: { width: 8, height: 8, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 0 4px var(--accent-soft)" },
  heroBio: { fontFamily: "var(--font-serif)", fontSize: "var(--t-xl)", lineHeight: 1.4, color: "var(--ink)", letterSpacing: "-0.005em" },
  heroBioP: { marginBottom: "var(--s-3)" },

  // Section header
  sec: { marginBottom: "var(--s-7)" },
  secHead: { display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "var(--s-4)" },
  secLeft: { display: "flex", alignItems: "baseline", gap: "var(--s-3)" },
  secNum: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)" },
  secTitle: { fontFamily: "var(--font-serif)", fontSize: "var(--t-xl)", fontWeight: 500, letterSpacing: "-0.01em" },
  secMeta: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em" },

  // Experience — compact dense rows with timeline gutter
  expWrap: { borderTop: "1px solid var(--rule-2)" },
  expRow: { display: "grid", gridTemplateColumns: "110px 36px 1fr auto", gap: "var(--s-3)", padding: "var(--s-3) 0", borderBottom: "1px solid var(--rule-2)", alignItems: "start" },
  expDate: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.04em", paddingTop: 4 },
  expLogo: { width: 36, height: 36, borderRadius: "var(--r-sm)", overflow: "hidden", background: "var(--paper-2)" },
  expRole: { fontWeight: 500, color: "var(--ink)", fontSize: "var(--t-sm)" },
  expOrg: { color: "var(--ink-2)", fontSize: "var(--t-sm)" },
  expNote: { color: "var(--ink-3)", fontSize: "var(--t-xs)", marginTop: 2 },
  expLoc: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", paddingTop: 4, textAlign: "right" },

  // Highlights — 4-up with tags
  hGrid: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "var(--s-3)" },
  hCard: { display: "block", border: "1px solid var(--rule-2)", borderRadius: "var(--r-md)", overflow: "hidden", background: "var(--paper-2)", transition: "transform .18s, box-shadow .18s" },
  hThumb: { aspectRatio: "1 / 1", background: "var(--paper-2)", borderBottom: "1px solid var(--rule-2)" },
  hMeta: { padding: "var(--s-2) var(--s-3) var(--s-3)" },
  hVenue: { fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.08em" },
  hTitle: { fontSize: "var(--t-sm)", marginTop: 4, fontWeight: 500, color: "var(--ink)" },

  // Publications — 2-col compact
  pubGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 var(--s-5)" },
  pubRow: { display: "grid", gridTemplateColumns: "60px 1fr", gap: "var(--s-3)", padding: "var(--s-3) 0", borderBottom: "1px solid var(--rule-2)", breakInside: "avoid" },
  pubThumb: { width: 60, height: 60, borderRadius: "var(--r-sm)", overflow: "hidden", background: "var(--paper-2)" },
  pubVenuePill: { display: "inline-block", fontFamily: "var(--font-mono)", fontSize: 10, padding: "2px 6px", background: "var(--accent-soft)", color: "var(--accent)", borderRadius: 3, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 },
  pubTitle: { fontSize: "var(--t-sm)", color: "var(--ink)", fontWeight: 500, lineHeight: 1.3 },
  pubAuthors: { fontSize: "var(--t-xs)", color: "var(--ink-3)", marginTop: 3 },
  pubLinks: { display: "flex", gap: 6, marginTop: 4, fontFamily: "var(--font-mono)", fontSize: 10 },
  pubLinkA: { color: "var(--ink-2)", borderBottom: "1px solid var(--rule)", paddingBottom: 1 },

  // Awards — two-col compact
  awardsGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 var(--s-5)" },
  awardLi: { display: "grid", gridTemplateColumns: "20px 1fr", gap: "var(--s-2)", padding: "6px 0", borderBottom: "1px dashed var(--rule-2)", fontSize: "var(--t-sm)", color: "var(--ink-2)" },
  awardOrn: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--accent)" },

  // Services — chip layout
  servBlock: { display: "grid", gridTemplateColumns: "100px 1fr", gap: "var(--s-3) var(--s-4)", padding: "var(--s-3) 0", borderBottom: "1px solid var(--rule-2)" },
  servLabel: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", color: "var(--ink-3)", textTransform: "uppercase", letterSpacing: "0.06em", paddingTop: 4 },
  chipWrap: { display: "flex", flexWrap: "wrap", gap: 4 },
  chip: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", padding: "3px 8px", background: "var(--paper-2)", borderRadius: 3, color: "var(--ink-2)" },

  // News — terminal-y log
  newsBlock: { fontFamily: "var(--font-mono)", fontSize: "var(--t-xs)", lineHeight: 1.7, padding: "var(--s-3) var(--s-4)", border: "1px solid var(--rule-2)", borderRadius: "var(--r-md)", background: "var(--paper-2)" },
  newsLine: { display: "grid", gridTemplateColumns: "90px 1fr", gap: "var(--s-3)", color: "var(--ink-2)" },
  newsDate: { color: "var(--accent)" },
};

function LabPlaceholder({ label, style }) {
  return <div className="cv-placeholder" style={style}>{label}</div>;
}

function LabCV() {
  const ls = LabStyles;

  const sections = [
    ["01", "Experience", "exp"],
    ["02", "Research", "research"],
    ["03", "Publications", "pubs"],
    ["04", "Awards", "awards"],
    ["05", "Services", "services"],
    ["06", "News", "news"],
  ];

  return (
    <div className="cv-reset" style={ls.shell}>
      {/* Side rail */}
      <aside style={ls.rail}>
        <div style={ls.railPhoto}>
          {RD.identity.photo
            ? <img src={RD.identity.photo} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            : <LabPlaceholder label="photo" style={{ width: "100%", height: "100%" }} />}
        </div>
        <div style={ls.railName}>{RD.identity.name}</div>
        <div style={ls.railTitle}>{RD.identity.title}</div>
        <div style={ls.railEmail}>{RD.identity.email}</div>

        <div style={ls.railNav}>
          {sections.map(([n, t, id]) => (
            <a key={id} href={`#${id}`} style={ls.navItem}>
              <span>{t}</span>
              <span style={ls.navItemNum}>{n}</span>
            </a>
          ))}
        </div>

        <div style={ls.railLinks}>
          {RD.links.map((l, i) => <a key={i} href={l.href} style={ls.railChip}>{l.label}</a>)}
        </div>
      </aside>

      {/* Main */}
      <main style={ls.main}>
        <header style={ls.hero}>
          <div style={ls.heroLine}>
            <span style={ls.heroDot}></span>
            <span>Available for collaboration</span>
            <span style={{ marginLeft: "auto", color: "var(--ink-3)" }}>{RD.identity.affiliation}</span>
          </div>
          <div style={ls.heroBio}>
            {RD.identity.bio.map((p, i) => <p key={i} style={ls.heroBioP}>{p}</p>)}
          </div>
        </header>

        {/* Experience */}
        <section id="exp" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>01</span>
              <h2 style={ls.secTitle}>Experience</h2>
            </div>
            <span style={ls.secMeta}>{RD.experience.length} positions</span>
          </div>
          <div style={ls.expWrap}>
            {RD.experience.map((e, i) => (
              <div key={i} style={ls.expRow}>
                <div style={ls.expDate}>{e.period}</div>
                <div style={ls.expLogo}>
                  {e.logo
                    ? <img src={e.logo} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    : <LabPlaceholder label="·" style={{ width: "100%", height: "100%", fontSize: 14 }} />}
                </div>
                <div>
                  <div style={ls.expRole}>{e.role}</div>
                  <div style={ls.expOrg}>{e.org}</div>
                  <div style={ls.expNote}>{e.note}</div>
                </div>
                <div style={ls.expLoc}>{e.location}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Highlights */}
        <section id="research" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>02</span>
              <h2 style={ls.secTitle}>Highlighted Research</h2>
            </div>
            <span style={ls.secMeta}>{RD.highlights.length} projects</span>
          </div>
          <div style={ls.hGrid}>
            {RD.highlights.map((h, i) => (
              <a key={i} href={h.href} style={ls.hCard}>
                <div style={ls.hThumb}>
                  {h.thumb
                    ? <img src={h.thumb} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    : <LabPlaceholder label="figure" style={{ width: "100%", height: "100%" }} />}
                </div>
                <div style={ls.hMeta}>
                  <div style={ls.hVenue}>{h.venue}</div>
                  <div style={ls.hTitle}>{h.title}</div>
                </div>
              </a>
            ))}
          </div>
        </section>

        {/* Publications */}
        <section id="pubs" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>03</span>
              <h2 style={ls.secTitle}>Publications</h2>
            </div>
            <span style={ls.secMeta}>{RD.publications.length} papers</span>
          </div>
          <div style={ls.pubGrid}>
            {RD.publications.map((p, i) => (
              <div key={i} style={ls.pubRow}>
                <div style={ls.pubThumb}>
                  {p.thumb
                    ? <img src={p.thumb} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    : <LabPlaceholder label="" style={{ width: "100%", height: "100%" }} />}
                </div>
                <div>
                  <div style={ls.pubVenuePill}>{p.venue} · {p.year}</div>
                  <div style={ls.pubTitle}>{p.title}</div>
                  <div style={ls.pubAuthors}>{p.authors}</div>
                  <div style={ls.pubLinks}>
                    {p.links.map((l, j) => <a key={j} href={l.href} style={ls.pubLinkA}>{l.label}</a>)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Awards */}
        <section id="awards" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>04</span>
              <h2 style={ls.secTitle}>Awards & Honors</h2>
            </div>
            <span style={ls.secMeta}>{RD.awards.length} entries</span>
          </div>
          <div style={ls.awardsGrid}>
            {RD.awards.map((a, i) => (
              <div key={i} style={ls.awardLi}>
                <span style={ls.awardOrn}>★</span>
                <span>{a}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Services */}
        <section id="services" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>05</span>
              <h2 style={ls.secTitle}>Academic Services</h2>
            </div>
          </div>
          <div style={ls.servBlock}>
            <div style={ls.servLabel}>Conferences</div>
            <div style={ls.chipWrap}>
              {RD.services.conferences.map((s, i) => <span key={i} style={ls.chip}>{s}</span>)}
            </div>
          </div>
          <div style={ls.servBlock}>
            <div style={ls.servLabel}>Journals</div>
            <div style={ls.chipWrap}>
              {RD.services.journals.map((s, i) => <span key={i} style={ls.chip}>{s}</span>)}
            </div>
          </div>
        </section>

        {/* News */}
        <section id="news" style={ls.sec}>
          <div style={ls.secHead}>
            <div style={ls.secLeft}>
              <span style={ls.secNum}>06</span>
              <h2 style={ls.secTitle}>Recent</h2>
            </div>
          </div>
          <div style={ls.newsBlock}>
            {RD.news.map((n, i) => (
              <div key={i} style={ls.newsLine}>
                <span style={ls.newsDate}>[{n.date}]</span>
                <span>{n.text}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

window.LabCV = LabCV;
