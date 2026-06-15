export const papers = [
  {
    id: 'nextlat',
    title: 'Next-Latent Prediction Transformers Learn Compact World Models',
    thumb: '/images/nextlat/nhs_manhattan.png',
    alt: 'Next-Latent Prediction',
    paperUrl: 'https://arxiv.org/abs/2511.05963',
    authors: [
      { name: 'Jayden Teoh', me: true },
      { name: 'Manan Tomar' },
      { name: 'Kwangjun Ahn' },
      { name: 'Edward S. Hu' },
      { name: 'Tim Pearce' },
      { name: 'Pratyusha Sharma' },
      { name: 'Akshay Krishnamurthy' },
      { name: 'Riashat Islam' },
      { name: 'Alex Lamb' },
      { name: 'John Langford' },
    ],
    venue: 'Microsoft Research Preprint',
    year: 2025,
    links: [
      { label: 'code', href: 'https://github.com/JaydenTeoh/NextLat' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2511.05963' },
      { label: 'blog post', href: 'https://jaydenteoh.github.io/2026/nextlat' },
    ],
    blurb:
      'We introduce Next-Latent Prediction (NextLat), which extends standard next-token training with self-supervised predictions in the latent space.',
  },
  {
    id: 'infogain',
    title: 'Improving Sampling for Masked Diffusion Models via Information Gain',
    thumb: '/images/infogain/illustrative.png',
    alt: 'Information Gain sampling',
    paperUrl: 'https://arxiv.org/abs/2602.18176',
    authors: [
      { name: 'Kaisen Yang' },
      { name: 'Jayden Teoh', me: true },
      { name: 'Kaicheng Yang' },
      { name: 'Yitong Zhang' },
      { name: 'Alex Lamb' },
    ],
    venue: 'ICML',
    year: 2026,
    links: [
      { label: 'code', href: 'https://github.com/yks23/Information-Gain-Sampler' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2602.18176' },
    ],
    blurb:
      'We introduce a decoding algorithm for Masked Diffusion Models (MDMs) that replaces greedy local-certainty heuristics with a principled information-gain objective, yielding more robust generation across math, code, and creative tasks.',
  },
  {
    id: 'dail',
    title: 'On Discovering Algorithms for Adversarial Imitation Learning',
    thumb: '/images/dail/Evo-IL.png',
    alt: 'Adversarial Imitation Learning',
    paperUrl: 'https://arxiv.org/abs/2510.00922',
    authors: [
      { name: 'Shashank Reddy Chirra' },
      { name: 'Jayden Teoh', me: true },
      { name: 'Praveen Paruchuri' },
      { name: 'Pradeep Varakantham' },
    ],
    venue: 'ICLR',
    year: 2026,
    links: [
      { label: 'code', href: 'https://github.com/shshnkreddy/DAIL' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2510.00922' },
    ],
    blurb:
      'We introduce a LLM-guided evolutionary framework for discovering new reward functions to stabilize Adversarial Imitation Learning.',
  },
  {
    id: 'morl-generalization',
    title: 'On Generalization Across Environments In Multi-Objective Reinforcement Learning',
    thumb: '/images/morl_generalization/square_envs.gif',
    alt: 'MORL generalization',
    paperUrl: 'https://arxiv.org/abs/2503.00799',
    authors: [
      { name: 'Jayden Teoh', me: true },
      { name: 'Pradeep Varakantham' },
      { name: 'Peter Vamplew' },
    ],
    venue: 'ICLR',
    year: 2025,
    links: [
      { label: 'code', href: 'https://github.com/JaydenTeoh/MORL-Generalization' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2503.00799' },
    ],
    blurb:
      'We formalize the concept of generalization in Multi-Objective Reinforcement Learning (MORL) and contribute a novel benchmark to facilitate future studies in this area.',
  },
  {
    id: 'elicitation-game',
    title: 'The Elicitation Game: Evaluating Capability Elicitation Techniques',
    thumb: '/images/elicitation_game/elicitation_figure.png',
    alt: 'The Elicitation Game',
    paperUrl: 'https://arxiv.org/abs/2502.02180',
    authors: [
      { name: 'Felix Hofstätter', equal: true },
      { name: 'Teun van der Weij', equal: true },
      { name: 'Jayden Teoh', me: true, equal: true },
      { name: 'Rada Djoneva' },
      { name: 'Henning Bartsch' },
      { name: 'Francis Rhys Ward' },
    ],
    venue: 'ICML',
    year: 2025,
    links: [
      { label: 'code', href: 'https://github.com/Felhof/sandbagging-elicitation' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2502.02180' },
      { label: 'twitter thread', href: 'https://x.com/Teun_vd_Weij/status/1895162797769793828' },
    ],
    blurb:
      'We evaluate the effectiveness of capability elicitation techniques by intentionally training language models with hidden capabilities that are revealed by a password.',
  },
  {
    id: 'cenie',
    title:
      'Improving Environment Novelty Quantification for Effective Unsupervised Environment Design',
    thumb: '/images/cenie/cenie_overview.png',
    alt: 'CENIE',
    paperUrl: 'https://arxiv.org/abs/2502.05726',
    authors: [
      { name: 'Jayden Teoh', me: true },
      { name: 'Wenjun Li' },
      { name: 'Pradeep Varakantham' },
    ],
    venue: 'NeurIPS',
    year: 2024,
    award: 'Oral Presentation',
    links: [
      { label: 'presentation', href: 'https://neurips.cc/virtual/2024/oral/97974' },
      { label: 'arXiv', href: 'https://arxiv.org/abs/2502.05726' },
    ],
    blurb:
      'CENIE proposes a framework for quantifying environment novelty in Unsupervised Environment Design to train agents that generalize better to unseen scenarios.',
  },
]
