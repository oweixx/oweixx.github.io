// ─────────────────────────────────────────────────────────────
// Sample data shared across all 3 template variants.
// All fields use {{TOKEN}} style placeholders so it's obvious
// what to replace. Edit this file (or replace with your own
// site/data.js when deploying) and every variant updates.
// ─────────────────────────────────────────────────────────────

window.CV_DATA = {
  identity: {
    name: "{{YOUR NAME}}",
    title: "{{ROLE — e.g. Ph.D. Candidate, Research Scientist}}",
    affiliation: "{{LAB / DEPARTMENT}}, {{INSTITUTION}}",
    email: "{{your.email@institution.edu}}",
    photo: "", // empty → striped placeholder; otherwise path to image
    bio: [
      "I am a {{ROLE}} at {{INSTITUTION}}.",
      "My research focuses on {{ONE-SENTENCE RESEARCH FOCUS}}. Specifically, I work on {{SPECIFIC PROBLEM AREA}}.",
    ],
  },

  links: [
    { label: "CV", href: "#" },
    { label: "Google Scholar", href: "#" },
    { label: "GitHub", href: "#" },
    { label: "LinkedIn", href: "#" },
    { label: "Twitter", href: "#" },
    { label: "Email", href: "mailto:{{your.email@institution.edu}}" },
  ],

  experience: [
    {
      org: "{{COMPANY / LAB}}",
      role: "{{Role — e.g. Research Scientist Intern}}",
      location: "{{City, Country / Remote}}",
      period: "{{Mon 'YY — Present}}",
      note: "{{Working on {{TOPIC}} (Manager: {{NAME}})}}",
      logo: "",
    },
    {
      org: "{{PREVIOUS COMPANY}}",
      role: "{{Role}}",
      location: "{{Location}}",
      period: "{{Mon 'YY — Mon 'YY}}",
      note: "{{Short description}}",
      logo: "",
    },
    {
      org: "{{INSTITUTION}}",
      role: "{{Role — e.g. Visiting Researcher}}",
      location: "{{Location}}",
      period: "{{Mon 'YY — Mon 'YY}}",
      note: "{{Short description}}",
      logo: "",
    },
    {
      org: "{{HOME INSTITUTION}}",
      role: "{{Ph.D. Candidate}}",
      location: "{{Location}}",
      period: "{{Mon 'YY — Mon 'YY (Expected)}}",
      note: "{{Advisor: {{NAME}}}}",
      logo: "",
    },
  ],

  highlights: [
    { title: "{{PROJECT 1}}", venue: "{{VENUE 'YY}}", thumb: "", href: "#" },
    { title: "{{PROJECT 2}}", venue: "{{VENUE 'YY}}", thumb: "", href: "#" },
    { title: "{{PROJECT 3}}", venue: "{{VENUE 'YY}}", thumb: "", href: "#" },
    { title: "{{PROJECT 4}}", venue: "{{VENUE 'YY}}", thumb: "", href: "#" },
  ],

  publications: [
    {
      year: "2026",
      title: "{{PAPER TITLE 1}}",
      authors: "{{Author A}}, {{Author B}}, {{YOU}}, {{Author C}}",
      venue: "{{Conference / Journal}}",
      links: [{ label: "Project", href: "#" }, { label: "PDF", href: "#" }, { label: "Code", href: "#" }],
      thumb: "",
    },
    {
      year: "2025",
      title: "{{PAPER TITLE 2}}",
      authors: "{{YOU}}, {{Author A}}, {{Author B}}",
      venue: "{{Conference}}",
      links: [{ label: "Project", href: "#" }, { label: "PDF", href: "#" }],
      thumb: "",
    },
    {
      year: "2024",
      title: "{{PAPER TITLE 3}}",
      authors: "{{YOU}}, {{Author A}}",
      venue: "{{Conference}}",
      links: [{ label: "PDF", href: "#" }, { label: "Code", href: "#" }],
      thumb: "",
    },
    {
      year: "2023",
      title: "{{PAPER TITLE 4}}",
      authors: "{{Author A}}, {{YOU}}",
      venue: "{{Workshop}}",
      links: [{ label: "PDF", href: "#" }],
      thumb: "",
    },
  ],

  awards: [
    "{{AWARD 1, ORGANIZATION, YEAR}}",
    "{{AWARD 2, ORGANIZATION, YEAR}}",
    "{{AWARD 3, ORGANIZATION, YEAR}}",
    "{{AWARD 4, ORGANIZATION, YEAR}}",
    "{{AWARD 5, ORGANIZATION, YEAR}}",
  ],

  services: {
    conferences: [
      "{{CVPR}}: {{2024, 2025}}",
      "{{ICCV}}: {{2023, 2025}}",
      "{{NeurIPS}}: {{2024, 2025}}",
    ],
    journals: [
      "{{TPAMI}}: {{2024, 2025}}",
      "{{IJCV}}: {{2024, 2025}}",
    ],
  },

  news: [
    { date: "{{YYYY-MM}}", text: "{{Recent update — paper accepted, talk, internship start, etc.}}" },
    { date: "{{YYYY-MM}}", text: "{{Earlier update}}" },
    { date: "{{YYYY-MM}}", text: "{{Earlier update}}" },
  ],
};
