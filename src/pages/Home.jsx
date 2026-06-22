import Header from '../components/Header.jsx'
import SEO from '../components/SEO.jsx'
import PaperCard from '../components/PaperCard.jsx'
import TypingText from '../components/TypingText.jsx'
import { papers } from '../data/papers.js'
import { siteDescription } from '../data/site.js'
import { writingLinks, academicService } from '../data/misc.js'

export default function Home() {
  return (
    <div className="app-shell">
      <SEO title="Jayden Teoh" description={siteDescription} path="/" />
      <Header />

      <main className="home">
        <section className="bio-section">
          <div className="bio-text">
            <h1 className="brand-name">
              <TypingText text="Jayden Teoh" />
            </h1>
            <p>
              I am currently a student researcher at{' '}
              <a href="https://deepmind.google/" target="_blank" rel="noreferrer">
                Google DeepMind
              </a>{' '}
              working with{' '}
              <a href="https://vaishnavh.github.io/" target="_blank" rel="noreferrer">
                Vaishnavh Nagarajan
              </a>
              . I will be joining MIT CSAIL as a PhD student in Fall 2026.
            </p>
            <p>
              Previously, I had a fun stint at{' '}
              <a
                href="https://www.microsoft.com/en-us/research/lab/microsoft-research-new-york/"
                target="_blank"
                rel="noreferrer"
              >
                Microsoft Research New York
              </a>{' '}
              under{' '}
              <a href="https://hunch.net/~jl/" target="_blank" rel="noreferrer">
                John Langford
              </a>{' '}
              where I worked on Next-Latent Prediction Transformers, a method for learning compact world models.
            </p>
            <p>I graduated with a Computer Science degree from Singapore Management University.</p>

            <p className="contact-row">
              <a href="mailto:t3ohjingxiang@gmail.com">email</a>
              <span className="sep">/</span>
              <a href="/data/Jayden_Academic_CV.pdf">cv</a>
              <span className="sep">/</span>
              <a href="https://www.linkedin.com/in/jayden-teoh/" target="_blank" rel="noreferrer">
                linkedin
              </a>
              <span className="sep">/</span>
              <a
                href="https://scholar.google.com/citations?user=GnHpLE8AAAAJ&hl=en"
                target="_blank"
                rel="noreferrer"
              >
                scholar
              </a>
              <span className="sep">/</span>
              <a href="https://x.com/jayden_teoh_" target="_blank" rel="noreferrer">
                x
              </a>
              <span className="sep">/</span>
              <a href="https://github.com/JaydenTeoh" target="_blank" rel="noreferrer">
                github
              </a>
            </p>
          </div>

          <div className="bio-portrait">
            <a href="/images/profile_pic_2.jpg">
              <img src="/images/profile_pic_2.jpg" alt="profile photo" />
            </a>
          </div>
        </section>

        <section className="section">
          <h2 className="section-heading">mentors I'm grateful for</h2>
          <ul className="plain-list">
            <li>
              <strong>Pradeep Varakantham (Singapore Management University)</strong> for taking a
              chance on me as an undergraduate and exposing me to the world of ML research. I'm
              forever grateful for his mentorship and support.
            </li>
            <li>
              <strong>John Langford (Microsoft Research)</strong> for providing me the space to
              grow as a researcher. His belief in my potential has been a driving force behind my
              decision to pursue a research career and a PhD.
            </li>
          </ul>
        </section>

        <section className="section">
          <h2 className="section-heading">research</h2>
          <p className="section-blurb">
            Jayden's research interests lie in the intersection of representation learning and
            self-improving agents. Jayden has a track record of publishing papers in top-tier
            conferences including NeurIPS, ICLR, and ICML.<sup>*</sup>
          </p>
          <p className="section-fineprint">
            *I've been told that writing in third person makes your credentials sound more legitimate :p
          </p>

          <div className="paper-list">
            {papers.map((p) => (
              <PaperCard key={p.id} paper={p} />
            ))}
          </div>
        </section>

        <section className="section">
          <h2 className="section-heading">miscellanea</h2>

          <div className="misc-row">
            <div className="misc-tag misc-tag-writing">writing</div>
            <ul className="plain-list misc-links">
              {writingLinks.map((w) => (
                <li key={w.href}>
                  <a href={w.href} target="_blank" rel="noreferrer">
                    {w.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div className="misc-row">
            <div className="misc-tag misc-tag-service">academic service</div>
            <ul className="plain-list misc-links">
              {academicService.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        </section>
      </main>
    </div>
  )
}
