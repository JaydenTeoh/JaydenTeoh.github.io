/** GPT-2 BPE token strings (leading spaces preserved). */
export const recipeGpt2Tokens = [
  'Recipe',
  ' says',
  ':',
  ' add',
  ' 2',
  ' tbsp',
  ' sugar',
  ',',
  ' stir',
  ' unt',
  'ill',
  ' dissolved',
  ',',
  ' then',
  ' put',
  ' in',
  ' the',
  ' fridge',
  ' for',
  ' 30',
  ' mins',
  ' lol',
]

/**
 * Benchmark slider slides. Set `src` to an image/gif path under /blog/assets/… when ready.
 * Optional: `mediaType: 'video'` for .mp4 (gifs use `src` + img).
 */
export const benchmarkSlides = [
  {
    id: 'world-modeling',
    label: 'World modeling',
    title: 'World modeling',
    description: 'We train the models on Manhattan taxi ride sequences. NextLat learns a world model that is not only more compact, but also more consistent with the real world!',
    src: '/blog/assets/2026-05-25-nextlat/manhattan_population_webby.gif',
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    title: 'Reasoning',
    description: 'NextLat achieves higher accuracy on the Countdown reasoning benchmark, which is notoriously difficult for LLMs (GPT-4 only gets 4%).',
    src: '/blog/assets/2026-05-25-nextlat/countdown.png',
  },
  {
    id: 'planning',
    label: 'Planning',
    title: 'Planning',
    description: 'NextLat is the only method capable of solving the Path-Star planning task, which is an unsolvable task for next-token prediction models.',
    src: '/blog/assets/2026-05-25-nextlat/path-star-perf.png',
  },
  {
    id: 'language-modeling',
    label: 'Language modeling',
    title: 'Language modeling',
    description: "NextLat's representations are more predictive of future tokens—up to 20 tokens ahead! NextLat also achieves the best downstream accuracy and lower perplexity than multi-token prediction methods in language modeling benchmarks.",
    src: '/blog/assets/2026-05-25-nextlat/nextlat_lm_results.png',
  },
]

export const transformerShortcutRefs = [
  {
    id: 'anil2022lengthgeneralization',
    href: 'https://arxiv.org/abs/2207.04901',
    authors: 'Anil et al.',
    year: 2022,
    title: 'Exploring Length Generalization in Large Language Models',
    venue: 'NeurIPS 2022',
  },
  {
    id: 'dziri2023faith',
    href: 'https://openreview.net/forum?id=Fkckkr3ya8',
    authors: 'Dziri et al.',
    year: 2023,
    title: 'Faith and Fate: Limits of Transformers on Compositionality',
    venue: 'NeurIPS 2023',
  },
  {
    id: 'liu2023shortcuts',
    href: 'https://openreview.net/forum?id=De4FYqjFueZ',
    authors: 'Liu et al.',
    year: 2023,
    title: 'Transformers Learn Shortcuts to Automata',
    venue: 'ICLR 2023',
  },
  {
    id: 'wu2024reasoning',
    href: 'https://aclanthology.org/2024.naacl-long.102/',
    authors: 'Wu et al.',
    year: 2024,
    title: 'Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks',
    venue: 'NAACL 2024',
  },
]

export const multiTokenPredictionRefs = [
  {
    id: 'gloeckle2024mtp',
    href: 'https://arxiv.org/abs/2404.19737',
    authors: 'Gloeckle et al.',
    year: 2024,
    title: 'Better & faster large language models via multi-token prediction',
    venue: 'arXiv:2404.19737',
  },
  {
    id: 'ahn2025joint',
    href: 'https://arxiv.org/abs/2503.21801',
    authors: 'Ahn et al.',
    year: 2025,
    title: 'Efficient joint prediction of multiple future tokens',
    venue: 'arXiv:2503.21801',
  },
  {
    id: 'hu2025belief',
    href: 'https://arxiv.org/abs/2410.23506',
    authors: 'Hu et al.',
    year: 2025,
    title: 'The belief state transformer',
    venue: 'ICLR 2025',
  },
  {
    id: 'shao2025beyond',
    href: 'https://openreview.net/forum?id=dDpB23VbVa',
    authors: 'Shao et al.',
    year: 2025,
    title: 'Beyond Next Token Prediction: Patch-Level Training for Large Language Models',
    venue: 'ICLR 2025',
  },
]


export const multiTokenPredictionModels = [
  {
    id: 'deepseekv3',
    href: 'https://arxiv.org/abs/2412.19437',
    authors: 'Liu et al.',
    year: 2024,
    title: 'Deepseek-v3 technical report',
    venue: 'arXiv:2412.19437',
  },
  {
    id: 'qwen2025qwen3next',
    href: 'https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd',
    authors: 'Qwen Team',
    year: 2025,
    title: 'Qwen3-Next: Towards Ultimate Training & Inference Efficiency',
    venue: 'Qwen Blog',
  },
  {
    id: 'nvidia2025nemotron3',
    href: 'https://arxiv.org/abs/2512.20856',
    authors: 'NVIDIA',
    year: 2025,
    title: 'NVIDIA Nemotron 3: Efficient and Open Intelligence',
    venue: 'White Paper',
  },
  {
    id: 'coreteam2026mimov2flashtechnicalreport',
    href: 'https://arxiv.org/abs/2601.02780',
    authors: 'Xiaomi MiMo Core Team et al.',
    year: 2026,
    title: 'MiMo-V2-Flash Technical Report',
    venue: 'arXiv:2601.02780',
  },
  {
    id: 'google2026gemma4mtp',
    href: 'https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/',
    authors: 'Google',
    year: 2026,
    title: 'Accelerating Gemma 4: faster inference with multi-token prediction drafters',
    venue: 'Google Blog',
  },
]
