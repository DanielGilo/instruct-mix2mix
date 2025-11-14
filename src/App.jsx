import React from "react";
import { Github, FileText, Video, Camera, Download, Link as LinkIcon, Mail, Copy } from "lucide-react";

// --------- Quick Start Notes ---------
// 1) This single-file React component is GitHub Pages–ready.
// 2) Create a new repo (e.g., `instruct-mix2mix`), drop this file in a Vite/Next/CRA app, or use the 
//    included minimal static export instructions in the footer.
// 3) Replace placeholder assets/links below with your own teaser images, method figures, and paper links.
// 4) Search for TODO to customize.

// -------------- Editable Data --------------
const META = {
  title: "InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization",
  tagline:
    "Sparse-view multi‑view editing via distilling a 2D editor into a multi‑view diffusion student for strong cross‑view consistency.",
  // TODO: replace with real links
  links: {
    paper: "#", // arXiv / OpenReview / camera-ready
    code: "#", // GitHub repo
    demo: "#", // project demo page or Hugging Face Space
    video: "#", // YouTube or Bilibili
    dataset: "#", // optional
  },
  // If anonymous for review, keep placeholders.
  authors: [
    // TODO: replace with real authors after de-anonymization
    { name: "Anonymous Authors", url: "#" },
  ],
  affiliations: ["—"],
  contactEmail: "your-email@example.com", // TODO
};

const ABSTRACT = `We tackle multi‑view image editing from only a few input views. InstructMix2Mix distills a powerful 2D
image editor into a pretrained multi‑view diffusion student, leveraging the student’s learned 3D prior to enforce cross‑view
coherence. Key ingredients include incremental student updates across denoising steps, a noise‑level schedule for the
teacher that avoids collapse, and a lightweight random cross‑view attention to couple views at virtually no extra cost.
Across diverse scenes and edits, the method improves multi‑view consistency while maintaining strong per‑frame edit quality.`;

// Gallery items (replace with your results). Use 3–8 images.
const TEASERS = [
  // TODO: replace with your own image URLs under /public or a CDN.
  { src: "https://picsum.photos/seed/elf/1200/700", caption: "Edit: \"Turn him into an Elf\" — multi‑view consistency." },
  { src: "https://picsum.photos/seed/knight/1200/700", caption: "Edit: \"Turn him into a knight\"." },
  { src: "https://picsum.photos/seed/facepaint/1200/700", caption: "Edit: \"Give him face paint\"." },
  { src: "https://picsum.photos/seed/panda/1200/700", caption: "Edit: \"Turn the bear to a panda\"." },
];

// Method figure (replace with your actual diagram)
const METHOD_FIG = {
  src: "https://picsum.photos/seed/method/1600/900",
  alt: "I‑Mix2Mix overview diagram",
  caption:
    "Overview. Distill edits from a 2D teacher into a multi‑view diffusion student with random cross‑view attention and a stable teacher noise schedule.",
};

// BibTeX (fill once de‑anonymized)
const BIBTEX = `@inproceedings{YourKey2026,
  title     = {InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization},
  author    = {To be updated},
  booktitle = {???},
  year      = {2026}
}`;

// -------------- UI Helpers --------------
function Pill({ children, href, icon: Icon }) {
  const content = (
    <span className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium shadow-sm ring-1 ring-black/10 hover:ring-black/20">
      {Icon ? <Icon className="h-4 w-4" /> : null}
      {children}
    </span>
  );
  if (!href || href === "#") return content;
  return (
    <a href={href} target="_blank" rel="noreferrer" className="transition-transform hover:-translate-y-0.5">
      {content}
    </a>
  );
}

function Section({ id, title, children, subtitle }) {
  return (
    <section id={id} className="mx-auto max-w-6xl px-6 py-16">
      <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">{title}</h2>
      {subtitle ? <p className="mt-2 text-neutral-600 dark:text-neutral-300">{subtitle}</p> : null}
      <div className="mt-8">{children}</div>
    </section>
  );
}

function CopyBox({ label, text }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <div className="relative rounded-xl border p-4 bg-white/70 dark:bg-black/30 backdrop-blur">
      <div className="absolute right-3 top-3">
        <button
          onClick={() => {
            navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="inline-flex items-center gap-1 rounded-md px-3 py-1 text-sm ring-1 ring-black/10 hover:ring-black/20"
        >
          <Copy className="h-4 w-4" /> {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="text-xs uppercase tracking-wide text-neutral-500 mb-2">{label}</div>
      <pre className="overflow-x-auto whitespace-pre-wrap text-sm leading-relaxed">{text}</pre>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t py-12 text-sm text-neutral-600 dark:text-neutral-300">
      <div className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="font-semibold">{META.title}</div>
            <div className="text-neutral-500">{META.venue}</div>
          </div>
          <div className="flex flex-wrap gap-3">
            <Pill href={META.links.code} icon={Github}>Code</Pill>
            <Pill href={META.links.paper} icon={FileText}>Paper</Pill>
            <Pill href={META.links.video} icon={Video}>Video</Pill>
            <Pill href={META.links.demo} icon={Camera}>Demo</Pill>
            {META.links.dataset && <Pill href={META.links.dataset} icon={Download}>Dataset</Pill>}
          </div>
        </div>
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <div className="rounded-xl border p-4">
            <div className="font-medium mb-2">How to host on GitHub Pages</div>
            <ol className="list-decimal ml-5 space-y-1 text-sm">
              <li>Create a new repo and initialize a React app (Vite recommended).</li>
              <li>Add this component as <code>src/App.jsx</code> (or import it there).</li>
              <li>In <code>package.json</code>, set <code>"homepage": "https://&lt;username&gt;.github.io/&lt;repo&gt;"</code>.</li>
              <li>Install <code>gh-pages</code> and add scripts: <code>predeploy</code> and <code>deploy</code>.</li>
              <li>Run <code>npm run deploy</code>; set Pages source to <code>gh-pages</code> branch.</li>
            </ol>
          </div>
          <div className="rounded-xl border p-4">
            <div className="font-medium mb-2">Attribution</div>
            <p className="text-sm leading-relaxed">
              Please cite our work if you find it useful. You can also link back to this page. For questions, reach out at
              {" "}
              <a href={`mailto:${META.contactEmail}`} className="inline-flex items-center gap-1 underline underline-offset-4">
                <Mail className="h-3.5 w-3.5" /> {META.contactEmail}
              </a>.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function ProjectPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-neutral-50 text-neutral-900 dark:from-neutral-950 dark:to-neutral-900 dark:text-neutral-50">
      {/* Hero */}
      <header className="relative overflow-hidden border-b">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(60rem_20rem_at_10%_-10%,rgba(0,0,0,0.05),transparent)] dark:bg-[radial-gradient(60rem_20rem_at_10%_-10%,rgba(255,255,255,0.08),transparent)]" />
        <div className="mx-auto max-w-6xl px-6 py-20">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div>
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">{META.title}</h1>
              <p className="mt-3 text-lg text-neutral-700 dark:text-neutral-300">{META.tagline}</p>
              <div className="mt-4 text-sm text-neutral-500">{META.venue}</div>
              <div className="mt-6 flex flex-wrap items-center gap-3 text-sm">
                {META.authors.map((a, i) => (
                  <a key={i} href={a.url} className="underline underline-offset-4" target="_blank" rel="noreferrer">
                    {a.name}
                  </a>
                ))}
                {META.affiliations?.length ? (
                  <span className="text-neutral-400">·</span>
                ) : null}
                <span className="text-neutral-500">{META.affiliations.join(", ")}</span>
              </div>
              <div className="mt-7 flex flex-wrap gap-3">
                <Pill href={META.links.paper} icon={FileText}>Paper</Pill>
                <Pill href={META.links.code} icon={Github}>Code</Pill>
                <Pill href={META.links.video} icon={Video}>Video</Pill>
                <Pill href={META.links.demo} icon={Camera}>Demo</Pill>
                {META.links.dataset && <Pill href={META.links.dataset} icon={Download}>Dataset</Pill>}
              </div>
            </div>
            <div className="w-full sm:w-[22rem] md:w-[26rem] lg:w-[30rem]">
              <div className="rounded-2xl border bg-white/60 p-4 shadow-sm backdrop-blur dark:bg-black/30">
                <div className="text-sm font-medium mb-1">Abstract</div>
                <p className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-200">{ABSTRACT}</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Teaser Gallery */}
      <Section id="results" title="Results" subtitle="Representative multi‑view edits. Replace these with your own before/after grids.">
        <div className="grid gap-6 md:grid-cols-2">
          {TEASERS.map((t, i) => (
            <figure key={i} className="overflow-hidden rounded-2xl border shadow-sm">
              <img src={t.src} alt={t.caption} className="w-full h-auto" />
              <figcaption className="p-3 text-sm text-neutral-600 dark:text-neutral-300">{t.caption}</figcaption>
            </figure>
          ))}
        </div>
      </Section>

      {/* Method */}
      <Section id="method" title="Method" subtitle="Distilling instruction‑following edits from a 2D teacher into a multi‑view diffusion student.">
        <figure className="overflow-hidden rounded-2xl border shadow-sm">
          <img src={METHOD_FIG.src} alt={METHOD_FIG.alt} className="w-full h-auto" />
          <figcaption className="p-3 text-sm text-neutral-600 dark:text-neutral-300">{METHOD_FIG.caption}</figcaption>
        </figure>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border p-4">
            <div className="font-semibold mb-1">Key ideas</div>
            <ul className="list-disc ml-5 space-y-1 text-sm">
              <li>Student = multi‑view diffusion model with a strong learned 3D prior.</li>
              <li>Distill from 2D teacher using incremental updates across denoising timesteps.</li>
              <li>Stochastic teacher noise schedule prevents degeneration / identity collapse.</li>
              <li>Random Cross‑View Attention couples frames at negligible overhead.</li>
            </ul>
          </div>
          <div className="rounded-xl border p-4">
            <div className="font-semibold mb-1">Why it works</div>
            <p className="text-sm leading-relaxed">
              The student’s 3D prior enforces geometry‑aware coherence, while the teacher contributes versatile instruction‑following edits.
              The SDS‑style objective backpropagates guidance into the student weights rather than aggressively altering latents, stabilizing training.
            </p>
          </div>
        </div>
      </Section>

      {/* Paper & BibTeX */}
      <Section id="paper" title="Paper & Resources">
        <div className="grid gap-6 md:grid-cols-3">
          <a href={META.links.paper} target="_blank" rel="noreferrer" className="rounded-xl border p-5 hover:shadow-sm">
            <div className="flex items-center gap-2 font-semibold"><FileText className="h-5 w-5"/> Paper</div>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-300">PDF, supplementary, and appendix.</p>
          </a>
          <a href={META.links.code} target="_blank" rel="noreferrer" className="rounded-xl border p-5 hover:shadow-sm">
            <div className="flex items-center gap-2 font-semibold"><Github className="h-5 w-5"/> Code</div>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-300">Training & inference, environment, checkpoints.</p>
          </a>
          <a href={META.links.video} target="_blank" rel="noreferrer" className="rounded-xl border p-5 hover:shadow-sm">
            <div className="flex items-center gap-2 font-semibold"><Video className="h-5 w-5"/> Video</div>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-300">Short method overview and results montage.</p>
          </a>
        </div>
        <div className="mt-8">
          <CopyBox label="BibTeX" text={BIBTEX} />
        </div>
      </Section>

      {/* FAQ */}
      <Section id="faq" title="FAQ">
        <div className="grid gap-4">
          <details className="rounded-xl border p-4">
            <summary className="cursor-pointer font-medium">What data do I need?</summary>
            <p className="mt-2 text-sm leading-relaxed">A sparse set of posed images (e.g., 4–8 views). If poses are not available, use COLMAP or an SfM pipeline to estimate them.</p>
          </details>
          <details className="rounded-xl border p-4">
            <summary className="cursor-pointer font-medium">How expensive is it?</summary>
            <p className="mt-2 text-sm leading-relaxed">We use single‑step student predictions, lightweight latent alignment, and random cross‑view attention to keep memory and compute moderate compared to extended‑attention baselines.</p>
          </details>
          <details className="rounded-xl border p-4">
            <summary className="cursor-pointer font-medium">Will you release code and weights?</summary>
            <p className="mt-2 text-sm leading-relaxed">Links will appear above upon release. Check back after the review period.</p>
          </details>
        </div>
      </Section>

      {/* Footer */}
      <Footer />

      {/* Floating links */}
      <div className="fixed bottom-6 right-6 flex flex-col gap-2">
        <a
          href={META.links.code}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 text-sm shadow-lg ring-1 ring-black/10 backdrop-blur hover:bg-white dark:bg-black/60"
        >
          <Github className="h-4 w-4" /> GitHub
        </a>
        <a
          href={META.links.paper}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 text-sm shadow-lg ring-1 ring-black/10 backdrop-blur hover:bg-white dark:bg-black/60"
        >
          <FileText className="h-4 w-4" /> PDF
        </a>
      </div>
    </main>
  );
}
