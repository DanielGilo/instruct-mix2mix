import React from "react";
import { Github, FileText, Camera, Mail, Copy } from "lucide-react";

// -------------- Editable Data --------------
const META = {
  title:
    "InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization",
  //venue: "Under review",
  links: {
    paper: "https://arxiv.org/abs/2511.14899", // arXiv / OpenReview / camera-ready
    code: "https://github.com/DanielGilo/instruct-mix2mix/tree/main", // GitHub repo
    demo: "#", // optional: demo / HF Space
  },
  authors: [
    {
      name: "Daniel Gilo",
      url: "https://scholar.google.com/citations?user=ARRwFY8AAAAJ&hl=en",
      affiliations: ["Technion — Israel Institute of Technology"],
      corresponding: true,
    },
    {
      name: "Or Litany",
      url: "https://orlitany.github.io/",
      affiliations: ["Technion — Israel Institute of Technology", "NVIDIA"],
    },
  ],
  contactEmail: "danielgilo@cs.technion.ac.il",
};

const ABSTRACT = (
  <>
    We address the task of <strong>multi-view image editing</strong> from sparse
    input views, where the inputs can be seen as a{" "}
    <span className="font-semibold">mix of images</span> capturing the scene
    from different viewpoints. The goal is to modify the scene according to a
    textual instruction while preserving consistency across all views. Existing
    methods, based on per-scene neural fields or temporal attention mechanisms,
    struggle in this setting, often producing artifacts and incoherent edits. We
    propose <strong>InstructMix2Mix (I-Mix2Mix)</strong>, a framework that
    distills the editing capabilities of a 2D diffusion model into a pretrained
    multi-view diffusion model, leveraging its data-driven 3D prior for
    cross-view consistency. A key contribution is{" "}
    <span className="font-semibold">
      replacing the conventional neural field consolidator in Score Distillation
      Sampling (SDS) with a multi-view diffusion student
    </span>
    , which requires novel adaptations: incremental student updates across
    timesteps, a specialized teacher noise scheduler to prevent degeneration,
    and an attention modification that enhances cross-view coherence without
    additional cost. Experiments demonstrate that I-Mix2Mix{" "}
    <span className="font-semibold">
      significantly improves multi-view consistency
    </span>{" "}
    while maintaining high per-frame edit quality.
  </>
);

// ------ Methods & Results (edit-by-edit comparison) ------

// Define the methods you want to compare.
const METHODS = [
  { key: "ours", label: "Ours" },
  { key: "igs2gs", label: "I-GS2GS" },
  { key: "t2vz", label: "T2VZ" },
  { key: "dge", label: "DGE" },
];

// For each edit, give per-method image paths.
const RESULTS = [
  {
    id: "ironman",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into Iron Man"',
    label: "Iron Man",
    defaultMethod: "ours",
    methods: {
      ours: "img/IronMan/im2m.png",
      igs2gs: "img/IronMan/igs2gs.png",
      t2vz: "img/IronMan/t2vz.png",
      dge: "img/IronMan/dge.png",
    },
  },
  {
    id: "knight",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into a knight"',
    label: "Knight",
    defaultMethod: "ours",
    methods: {
      ours: "img/Knight/im2m.png",
      igs2gs: "img/Knight/igs2gs.png",
      t2vz: "img/Knight/t2vz.png",
      dge: "img/Knight/dge.png",
    },
  },
  {
    id: "clown",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into a clown"',
    label: "Clown",
    defaultMethod: "ours",
    methods: {
      ours: "img/Clown/im2m.png",
      igs2gs: "img/Clown/igs2gs.png",
      t2vz: "img/Clown/t2vz.png",
      dge: "img/Clown/dge.png",
    },
  },
  {
    id: "robot",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into a robot"',
    label: "Robot",
    defaultMethod: "ours",
    methods: {
      ours: "img/Robot/im2m.png",
      igs2gs: "img/Robot/igs2gs.png",
      t2vz: "img/Robot/t2vz.png",
      dge: "img/Robot/dge.png",
    },
  },
  {
    id: "marble_statue",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into a marble statue"',
    label: "Marble Statue",
    defaultMethod: "ours",
    methods: {
      ours: "img/MarbleStatue/im2m.png",
      igs2gs: "img/MarbleStatue/igs2gs.png",
      t2vz: "img/MarbleStatue/t2vz.png",
      dge: "img/MarbleStatue/dge.png",
    },
  },
  {
    id: "soldier",
    sceneId: "scene1",
    sceneName: "Scene 1",
    originalSrc: "img/person-original.png",
    title: '"Turn him into a soldier"',
    label: "Soldier",
    defaultMethod: "ours",
    methods: {
      ours: "img/Soldier/im2m.png",
      igs2gs: "img/Soldier/igs2gs.png",
      t2vz: "img/Soldier/t2vz.png",
      dge: "img/Soldier/dge.png",
    },
  },
  {
    id: "face_paint",
    sceneId: "scene2",
    sceneName: "Scene 2",
    originalSrc: "img/face-original.png",
    title: '"Give him face paint"',
    label: "Face Paint",
    defaultMethod: "ours",
    methods: {
      ours: "img/FacePaint/im2m.png",
      igs2gs: "img/FacePaint/igs2gs.png",
      t2vz: "img/FacePaint/t2vz.png",
      dge: "img/FacePaint/dge.png",
    },
  },
  {
    id: "skull",
    sceneId: "scene2",
    sceneName: "Scene 2",
    originalSrc: "img/face-original.png",
    title: '"Turn his face into a skull"',
    label: "Skull",
    defaultMethod: "ours",
    methods: {
      ours: "img/Skull/im2m.png",
      igs2gs: "img/Skull/igs2gs.png",
      t2vz: "img/Skull/t2vz.png",
      dge: "img/Skull/dge.png",
    },
  },
  {
    id: "vampire",
    sceneId: "scene2",
    sceneName: "Scene 2",
    originalSrc: "img/face-original.png",
    title: '"Turn him into a vampire"',
    label: "Vampire",
    defaultMethod: "ours",
    methods: {
      ours: "img/Vampire/im2m.png",
      igs2gs: "img/Vampire/igs2gs.png",
      t2vz: "img/Vampire/t2vz.png",
      dge: "img/Vampire/dge.png",
    },
  },
  {
    id: "tolkein_elf",
    sceneId: "scene2",
    sceneName: "Scene 2",
    originalSrc: "img/face-original.png",
    title: '"Turn him into a Tolkein Elf"',
    label: "Tolkien Elf",
    defaultMethod: "ours",
    methods: {
      ours: "img/TolkeinElf/im2m.png",
      igs2gs: "img/TolkeinElf/igs2gs.png",
      t2vz: "img/TolkeinElf/t2vz.png",
      dge: "img/TolkeinElf/dge.png",
    },
  },
  {
    id: "panda",
    sceneId: "scene3",
    sceneName: "Scene 3",
    originalSrc: "img/bear-original.png",
    title: '"Turn the bear to a panda bear"',
    label: "Panda Bear",
    defaultMethod: "ours",
    methods: {
      ours: "img/PandaBear/im2m.png",
      igs2gs: "img/PandaBear/igs2gs.png",
      t2vz: "img/PandaBear/t2vz.png",
      dge: "img/PandaBear/dge.png",
    },
  },
  {
    id: "polar",
    sceneId: "scene3",
    sceneName: "Scene 3",
    originalSrc: "img/bear-original.png",
    title: '"Turn the bear to a Polar bear"',
    label: "Polar Bear",
    defaultMethod: "ours",
    methods: {
      ours: "img/PolarBear/im2m.png",
      igs2gs: "img/PolarBear/igs2gs.png",
      t2vz: "img/PolarBear/t2vz.png",
      dge: "img/PolarBear/dge.png",
    },
  },
  {
    id: "grizzly",
    sceneId: "scene3",
    sceneName: "Scene 3",
    originalSrc: "img/bear-original.png",
    title: '"Turn the bear to a Grizzly bear"',
    label: "Grizzly Bear",
    defaultMethod: "ours",
    methods: {
      ours: "img/GrizzlyBear/im2m.png",
      igs2gs: "img/GrizzlyBear/igs2gs.png",
      t2vz: "img/GrizzlyBear/t2vz.png",
      dge: "img/GrizzlyBear/dge.png",
    },
  },
];

// Method figure
const METHOD_FIG = {
  src: "img/method-figure.png",
  alt: "I-Mix2Mix overview diagram",
  caption: (
    <>
      Given a set of input images, we randomly choose a reference frame, edit it
      with the frozen teacher, and encode it to initialize the personalized
      multi-view student (<em>Initialization</em>). At each distillation
      iteration, noisy multi-view latents are denoised by the student (
      <em>Student Query</em>), aligned to the teacher’s latent space (
      <em>Alignment</em>), and perturbed with our forward schedule (
      <em>Perturbation</em>). The teacher predicts edits with Random
      Cross-View Attention (<em>Teacher Prediction</em>), where all frames
      attend to the κ-frame, and the resulting supervision is distilled back
      into the student (<em>Student Update</em>). After distillation, the
      student outputs a set of multi-view-consistent edited frames.
    </>
  ),
};

// Teaser figure
const TEASER_FIG = {
  src: "img/teaser-figure.png",
};

// References
const REFERENCES = [
  "Cyrus Vachha and Ayaan Haque. Instruct-gs2gs: Editing 3d gaussian splats with instructions. 2024.",
  "Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. ICCV, 2023.",
  "Minghao Chen, Iro Laina, and Andrea Vedaldi. DGE: Direct Gaussian 3D Editing by Consistent Multi-View Editing. ECCV 2024.",
];

// BibTeX (fill once de-anonymized)
const BIBTEX = `@misc{gilo2025instructmix2mixconsistentsparseviewediting,
      title={InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization}, 
      author={Daniel Gilo and Or Litany},
      year={2025},
      eprint={2511.14899},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14899}, 
}`;

// -------------- UI Helpers --------------
function Pill({ children, href, icon: Icon }) {
  const content = (
    <span className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium bg-white text-slate-800 shadow-sm ring-1 ring-slate-200 hover:ring-slate-300">
      {Icon ? <Icon className="h-4 w-4" /> : null}
      {children}
    </span>
  );
  if (!href || href === "#") return content;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="transition-transform hover:-translate-y-0.5"
    >
      {content}
    </a>
  );
}

function Section({ id, title, children, subtitle }) {
  return (
    <section id={id} className="mx-auto max-w-6xl px-6 py-16">
      <div className="mb-8">
        <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-900">
          {title}
        </h2>
        {subtitle ? (
          <p className="mt-2 max-w-2xl text-sm sm:text-base text-slate-600">
            {subtitle}
          </p>
        ) : null}
      </div>
      {children}
    </section>
  );
}

function CopyBox({ label, text }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <div className="relative rounded-2xl border border-slate-200 bg-white/90 p-4 shadow-sm">
      <div className="absolute right-3 top-3">
        <button
          onClick={() => {
            navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200 hover:ring-slate-300 bg-white"
        >
          <Copy className="h-3.5 w-3.5" /> {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <pre className="overflow-x-auto whitespace-pre-wrap text-xs sm:text-sm leading-relaxed text-slate-800">
        {text}
      </pre>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white py-10 text-sm text-slate-600">
      <div className="mx-auto max-w-6xl px-6">
        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
          <div className="mb-1 text-sm font-semibold text-slate-900">
            Attribution
          </div>
          <p className="text-sm leading-relaxed text-slate-600">
            Please cite our work if you find it useful. For questions, reach out
            at{" "}
            <a
              href={`mailto:${META.contactEmail}`}
              className="inline-flex items-center gap-1 font-medium text-slate-900 underline underline-offset-4"
            >
              <Mail className="h-3.5 w-3.5" /> {META.contactEmail}
            </a>
            .
          </p>
        </div>
      </div>
    </footer>
  );
}

// -------------- Scene Card (original + edits + methods) --------------
function SceneCard({ scene }) {
  const [activeEditId, setActiveEditId] = React.useState(scene.edits[0]?.id);
  const activeEdit = scene.edits.find((e) => e.id === activeEditId);

  const [activeMethod, setActiveMethod] = React.useState(
    activeEdit?.defaultMethod || "ours",
  );

  React.useEffect(() => {
    if (activeEdit) {
      setActiveMethod(activeEdit.defaultMethod || "ours");
    }
  }, [activeEdit?.id]);

  const availableMethods = activeEdit
    ? METHODS.filter((m) => activeEdit.methods[m.key])
    : [];
  const activeImageSrc = activeEdit ? activeEdit.methods[activeMethod] : null;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 lg:p-5 shadow-sm">
      <div className="mb-3 flex justify-end">
        <span className="text-[11px] uppercase tracking-wide text-slate-400">
          Original &amp; edited views
        </span>
      </div>

      {/* Frames row: original vs edited */}
      <div className="grid gap-4 md:grid-cols-2 items-stretch">
        <figure className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
          {scene.originalSrc && (
            <img
              src={scene.originalSrc}
              alt={`${scene.name} – original views`}
              className="h-auto w-full"
            />
          )}
          <figcaption className="px-3 py-2 text-[11px] text-slate-500">
            Original input frames.
          </figcaption>
        </figure>

        <figure className="flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
          {activeImageSrc ? (
            <img
              src={activeImageSrc}
              alt={`${activeEdit?.label || activeEdit?.title} – ${activeMethod}`}
              className="h-auto w-full"
            />
          ) : (
            <div className="flex flex-1 items-center justify-center px-4 py-6 text-xs text-slate-500">
              Add an image for method <code>{activeMethod}</code> in the config.
            </div>
          )}
          <figcaption className="px-3 py-2 text-[11px] text-slate-500">
            Edited frames ({activeEdit?.label || activeEdit?.title}).
          </figcaption>
        </figure>
      </div>

      {/* Controls row: prompt + edits + methods */}
      {activeEdit && (
        <div className="mt-4 space-y-3">
          {activeEdit.title && (
            <div className="inline-flex items-center gap-1 rounded-full bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-600 ring-1 ring-slate-200">
              <span className="text-slate-400">Prompt:</span>{" "}
              <span className="italic">{activeEdit.title}</span>
            </div>
          )}

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            {/* Edit selector */}
            {scene.edits.length > 1 && (
              <div className="flex-1">
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Edits
                </div>
                <div className="flex flex-wrap gap-1 text-xs">
                  {scene.edits.map((edit) => {
                    const isActive = edit.id === activeEditId;
                    return (
                      <button
                        key={edit.id}
                        type="button"
                        onClick={() => setActiveEditId(edit.id)}
                        className={
                          "rounded-full px-3 py-1 font-medium transition-colors " +
                          (isActive
                            ? "bg-slate-900 text-slate-50"
                            : "bg-slate-100 text-slate-700 hover:bg-white ring-1 ring-slate-200")
                        }
                      >
                        {edit.label || edit.title}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Method selector */}
            <div className="flex-1">
              <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-slate-500">
                Methods
              </div>
              <div className="flex flex-wrap gap-1 text-xs">
                {availableMethods.map((m) => {
                  const isActive = m.key === activeMethod;
                  return (
                    <button
                      key={m.key}
                      type="button"
                      onClick={() => setActiveMethod(m.key)}
                      className={
                        "rounded-full px-3 py-1 font-medium transition-colors " +
                        (isActive
                          ? "bg-slate-900 text-slate-50"
                          : "bg-slate-100 text-slate-700 hover:bg-white ring-1 ring-slate-200")
                      }
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// -------------- Page --------------
export default function ProjectPage() {
  // Group RESULTS by scene
  const sceneMap = {};
  RESULTS.forEach((ex) => {
    const id = ex.sceneId || ex.id;
    if (!sceneMap[id]) {
      sceneMap[id] = {
        id,
        name: ex.sceneName || `Scene ${id}`,
        originalSrc: ex.originalSrc,
        edits: [],
      };
    }
    sceneMap[id].edits.push(ex);
  });
  const scenes = Object.values(sceneMap);

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-slate-50 text-slate-900">
      {/* Hero */}
      <header className="relative border-b border-slate-200 bg-white/90">
        <div className="pointer-events-none absolute inset-x-0 top-[-6rem] -z-10 mx-auto h-72 max-w-3xl bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.18),_transparent_65%)]" />
        <div className="mx-auto flex max-w-6xl flex-col gap-10 px-6 py-16 sm:py-20 lg:flex-row lg:items-start lg:justify-between">
          {/* Left: title, authors, buttons */}
          
          <div className="max-w-xl space-y-6">
            {META.venue && (
              <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                {META.venue}
              </div>
              )}
            <div>
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight text-slate-900">
                {META.title}
              </h1>
              {/* TL;DR */}
              <p className="mt-3 text-sm sm:text-base text-slate-700">
                <span className="font-semibold">TL;DR:</span> I-Mix2Mix performs
                instruction-driven edits on a{" "}
                <span className="font-semibold italic">sparse set of views</span>.
                The key idea is <strong>SDS with a twist</strong>: we distill a 2D
                editor into a{" "}
                <span className="font-semibold italic">
                  pretrained multi-view diffusion
                </span>{" "}
                model rather than a NeRF/3DGS. The student’s learned 3D prior
                enables multi-view consistent edits, despite the sparse input.
              </p>
            </div>

            {/* Authors */}
            <div className="space-y-1 text-sm">
              {META.authors.map((a, i) => (
                <div key={i} className="flex flex-wrap items-center gap-2">
                  {a.url && a.url !== "#" ? (
                    <>
                      <a
                        href={a.url}
                        target="_blank"
                        rel="noreferrer"
                        className="font-medium text-slate-900 underline underline-offset-4"
                      >
                        {a.name}
                      </a>
                      {a.corresponding && (
                        <sup
                          className="text-[10px] text-slate-500"
                          title="Corresponding author"
                        >
                          †
                        </sup>
                      )}
                    </>
                  ) : (
                    <>
                      <span className="font-medium text-slate-900">
                        {a.name}
                      </span>
                      {a.corresponding && (
                        <sup
                          className="text-[10px] text-slate-500"
                          title="Corresponding author"
                        >
                          †
                        </sup>
                      )}
                    </>
                  )}
                  {a.affiliations?.map((af, j) => (
                    <span
                      key={j}
                      className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700 ring-1 ring-slate-200"
                    >
                      {af}
                    </span>
                  ))}
                </div>
              ))}
              <p className="mt-1 text-xs text-slate-500">
                † Corresponding author.
              </p>
            </div>

            {/* Primary actions */}
            <div className="flex flex-wrap gap-3 pt-2">
              <Pill href={META.links.paper} icon={FileText}>
                Paper
              </Pill>
              <Pill href={META.links.code} icon={Github}>
                Code
              </Pill>
              {META.links.demo && META.links.demo !== "#" && (
                <Pill href={META.links.demo} icon={Camera}>
                  Demo
                </Pill>
              )}
            </div>
          </div>

          {/* Right: abstract card */}
          <div className="w-full max-w-md lg:pt-6">
            <div className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Abstract
              </div>
              <p className="text-sm leading-relaxed text-slate-700">
                {ABSTRACT}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Teaser */}
      <Section id="teaser" title="Sparse Multi-View Editing Examples">
        <figure className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <img
            src={TEASER_FIG.src}
            alt={TEASER_FIG.alt}
            className="h-auto w-full"
          />
          <figcaption className="p-3 text-xs sm:text-sm text-slate-600">
            {TEASER_FIG.caption}
          </figcaption>
        </figure>
      </Section>

      {/* Method */}
<Section
  id="method"
  title="Method"
  subtitle="Distilling instruction-following edits from a 2D teacher into a multi-view diffusion student."
>
  <div className="space-y-6">
    <figure className="overflow-hidden rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
      <img
        src={METHOD_FIG.src}
        alt={METHOD_FIG.alt}
        className="h-auto w-full rounded-xl"
      />
      <figcaption className="p-3 text-[13px] sm:text-sm text-slate-600">
        {METHOD_FIG.caption}
      </figcaption>
    </figure>

    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="mb-1 text-base font-semibold text-slate-900">
        Core Ideas Behind I-Mix2Mix
      </div>

      <p className="text-base leading-relaxed text-slate-600">
        Editing sparse multi-view inputs is challenging: prior methods rely on
        dense coverage and often produce artifacts or inconsistent edits when
        views are limited. Our approach compensates for the missing viewpoints
        by leveraging a <span className="font-semibold">strong 3D prior</span>—a pretrained,
        state-of-the-art <span className="font-semibold">multi-view diffusion model</span> that
        serves as our student. We then distill
        <span className="font-semibold"> instruction-following editing ability</span> from a
        2D editing teacher using an SDS-like objective.
      </p>

      <p className="mt-3 text-base leading-relaxed text-slate-600">
        Replacing a neural field with a multi-view diffusion model inside SDS is
        non-trivial and requires several key adaptations:
      </p>

      <ol className="mt-2 ml-5 list-decimal space-y-1 text-base text-slate-600">
        <li>
          Instead of rendering from an explicit scene representation, we sample
          views directly from the diffusion student. To avoid costly full
          diffusion trajectories, we{" "}
          <span className="font-semibold">
            distill incrementally across student timesteps
          </span>.
        </li>
        <li>
          We introduce a{" "}
          <span className="font-semibold">specialized teacher noise scheduler</span>{" "}
          that stabilizes supervision and prevents collapse to poor local
          minima.
        </li>
        <li>
          A lightweight{" "}
          <span className="font-semibold">cross-view attention coupling</span>{" "}
          improves multi-view consistency at no additional computational cost.
        </li>
      </ol>

      <p className="mt-3 text-base leading-relaxed text-slate-600">
        Together, these components enable stable, high-quality, and
        3D-consistent edits from extremely sparse inputs—something prior
        approaches consistently struggle to achieve.
      </p>
    </div>
  </div>
</Section>



      {/* Comparison with other methods */}
      <Section
        id="comparison"
        title="Comparison to Other Methods"
        subtitle={
          <>
            Compare edits of <span className="font-semibold">I-Mix2Mix</span>{" "}
            and popular baselines: Instruct-GS2GS (I-GS2GS) [1], Text2Video-Zero (T2VZ) [2], Direct Gaussian Editing (DGE) [3].
            Matching{" "}
            <span className="font-semibold text-red-500">red</span> and{" "}
            <span className="font-semibold text-purple-500">purple</span>{" "}
            rectangles indicate inconsistent regions, which frequently appear in
            baselines but rarely in our edits.
          </>
        }
      >
        <div className="space-y-6">
          {scenes.map((scene) => (
            <SceneCard key={scene.id} scene={scene} />
          ))}
        </div>
      </Section>

      <Section id="references" title="BibTeX & References">
        <CopyBox label="BibTeX" text={BIBTEX} />

        <div className="mt-8">
          <ol className="list-decimal space-y-1 pl-5 text-sm text-slate-700">
            {REFERENCES.map((ref, idx) => (
              <li key={idx}>{ref}</li>
            ))}
          </ol>
        </div>
      </Section>


      <Footer />

      {/* Floating shortcuts */}
      <div className="fixed bottom-6 right-6 flex flex-col gap-2">
        <a
          href={META.links.code}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs font-medium text-slate-800 shadow-lg ring-1 ring-slate-200 hover:ring-slate-300"
        >
          <Github className="h-4 w-4" /> GitHub
        </a>
        <a
          href={META.links.paper}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-xs font-medium text-slate-800 shadow-lg ring-1 ring-slate-200 hover:ring-slate-300"
        >
          <FileText className="h-4 w-4" /> PDF
        </a>
      </div>
    </main>
  );
}
