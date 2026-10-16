import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";

const project = { name: "AutoPlay QA", repositoryUrl: "https://github.com/ShaunNYC/autoplay-qa" };
const author = {
  name: "Shaun Wang",
  github: "https://github.com/ShaunNYC",
  linkedin: "https://www.linkedin.com/in/shaun-wang-3b92222b8",
};

export function ProjectIntro() {
  return <section className="project-intro" aria-labelledby="project-title">
    <div className="project-identity">
      <a href={project.repositoryUrl} target="_blank" rel="noopener noreferrer">GitHub / {project.name}<ArrowUpRight size={22} /></a>
    </div>
    <div className="intro-composition">
      <div className="intro-copy">
        <h1 id="project-title"><span>Self-Evolving</span><span className="title-accent">Video Game</span><span className="title-accent">QA</span></h1>
        <div className="intro-actions"><a className="button hero-button" href="#evidence-workspace">Explore the evidence<ArrowDownRight size={22} /></a><a className="intro-secondary" href="#the-method">About the project<ArrowRight size={20} /></a></div>
      </div>
      <div className="evidence-poster" aria-label="Illustration of the evidence workflow">
        <h2 className="poster-caption">The evidence workflow<ArrowUpRight size={25} /></h2>
        <div className="poster-diagram" aria-hidden="true">
          <div className="diagram-grid"><span className="diagram-wall wall-one" /><span className="diagram-wall wall-two" /><span className="diagram-wall wall-three" /><span className="diagram-door" /><svg viewBox="0 0 320 230"><path d="M40 185H115V125H205V55H270" /><circle cx="40" cy="185" r="7" /><circle className="diagram-target" cx="270" cy="55" r="13" /></svg></div>
          <span className="proof-stamp"><span>Follow<br />the trail</span><ArrowDownRight size={25} /></span>
        </div>
        <div className="poster-receipt"><strong>Control → Observe → Review</strong></div>
      </div>
    </div>
  </section>;
}

export function ProjectMethod() {
  return <section className="project-method" id="the-method" aria-labelledby="method-title">
    <div className="method-heading"><h2 id="method-title">An open workbench.<br /><span>A self-evolving vision.</span></h2><p>AutoPlay QA connects NVIDIA Nemotron to recorded game feedback, native replay and developer review. Our vision is AI testers that improve through experience.</p></div>
    <div className="method-steps">
      <article><h3>Play within bounds.</h3><p>Nemotron Lightning chooses controls from visible game feedback. Actions, requests, time and estimated spend have explicit limits.</p></article>
      <article><h3>Keep every source.</h3><p>Nemotron Ultra reviews recorded evidence with step citations. Software signals, AI judgments and human review stay distinguishable.</p></article>
      <article><h3>Make it replayable.</h3><p>Inspect before and after states, export reports and replay controls in the original runtime. Developers can check the claim themselves.</p></article>
    </div>
  </section>;
}

export function CommunityDirections() {
  return <section className="community-directions" aria-labelledby="community-title">
    <div className="community-heading"><h2 id="community-title">Better testers.<br /><span>Built together.</span></h2><p>A shared foundation for exploring how AI can investigate games. The next step is a community effort, with inspectable evidence at its core.</p></div>
    <div className="direction-list">
      <article><h3>More game interfaces</h3><p>Grid and Unity adapters form the current showcase. An experimental Mars browser adapter explores editor workflows.</p></article>
      <article><h3>Tests that start with a question</h3><p>An experimental protocol records hypotheses, predictions and control experiments before actions, with regression export tooling.</p></article>
      <article><h3>Experience worth keeping</h3><p>Archived context and memory prototypes provide a starting point for community research into stronger investigation and learning.</p></article>
    </div>
  </section>;
}

export function AboutAuthor() {
  return <section className="about-author" aria-labelledby="author-title">
    <h2 id="author-title">About the author</h2>
    <div><h3>{author.name}</h3><p>I’m building an open workbench for AI game testing, with evidence developers can inspect and replay.</p><div className="author-links"><a href={author.github} target="_blank" rel="noopener noreferrer">GitHub<ArrowUpRight size={22} /></a><a href={author.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn<ArrowUpRight size={22} /></a></div></div>
  </section>;
}
