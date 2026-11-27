const PRESETS = {
  shuidiao: {
    lora: "LoRA: classics-poetry (18L Rank-8 · 1.0x)",
    initialCommitted: "",
    preedit: "ming'yue'ji'shi'you",
    candidates: [
      { text: "明月几时有", tag: "⇥ 诗句" },
      { text: "明月即使有", tag: "" },
      { text: "明月记事有", tag: "" }
    ],
    chain: [
      "，把酒问青天",
      "。不知天上宫阙",
      "，今夕是何年",
      "。我欲乘风归去",
      "，又恐琼楼玉宇",
      "，高处不胜寒",
      "。起舞弄清影",
      "，何似在人间",
      "。人有悲欢离合",
      "，月有阴晴圆缺",
      "，此事古难全",
      "。但愿人长久",
      "，千里共婵娟。"
    ]
  },
  jiangjinjiu: {
    lora: "LoRA: classics-poetry (18L Rank-8 · 防重复消歧)",
    initialCommitted: "君不见黄河之水天上来，奔流到海不复回。",
    preedit: "jun'bu'jian",
    candidates: [
      { text: "君不见高堂明镜悲白发", tag: "⇥ 下句晋升" },
      { text: "君不见", tag: "" },
      { text: "君不见黄河之水天上来", tag: "已输降权" }
    ],
    chain: [
      "，朝如青丝暮成雪",
      "。人生得意须尽欢",
      "，莫使金樽空对月",
      "。天生我材必有用",
      "，千金散尽还复来",
      "。烹羊宰牛且为乐",
      "，会须一饮三百杯。"
    ]
  },
  yulinling: {
    lora: "LoRA: classics-poetry (18L Rank-8 · 乱序纠错 uq→qu)",
    initialCommitted: "执手相看泪眼，竟无语凝噎。",
    preedit: "nian'qu'uq'qian'li'yan'bo",
    candidates: [
      { text: "念去去，千里烟波", tag: "纠错+领字连读" },
      { text: "念区区千里烟波", tag: "" }
    ],
    chain: [
      "，暮霭沉沉楚天阔",
      "。多情自古伤离别",
      "，更那堪，冷落清秋节",
      "！今宵酒醒何处",
      "？杨柳岸，晓风残月。"
    ]
  },
  codeswitch: {
    lora: "LoRA: tech-developer (18L Rank-8 · 中/En 双模式英文纠错)",
    initialCommitted: "",
    preedit: "bang'wo'reivew'zhe'ge'pr",
    candidates: [
      { text: "帮我 review 这个 PR", tag: "自动纠错 reivew→review" },
      { text: "帮我 reivew 这个 PR", tag: "原词保留" }
    ],
    chain: []
  },
  singlegen: {
    lora: "Route A · Qwen3 BI-18L e8_l4 MLX (190.3 MB · 36K Full-CJK)",
    initialCommitted: "",
    preedit: "gen",
    candidates: [
      { text: "跟", tag: "" },
      { text: "根", tag: "" },
      { text: "根据", tag: "高频双字" },
      { text: "根本", tag: "高频双字" },
      { text: "更", tag: "模糊音" }
    ],
    chain: []
  }
};

const I18N_EN = {
  "nav.demo": "Live Demo",
  "nav.arch": "Architecture",
  "nav.bench": "Benchmarks",
  "nav.packs": "LoRA Packs",
  "nav.diy": "Private LoRA",
  "hero.eyebrow": "macOS 15+ · Apple Silicon · Apple Notarized · 100% Offline · MIT / Apache-2.0",
  "hero.title": "BI-18L Mixed-Precision Qwen3 in Unified Memory — 25% smaller, 3× faster, 80.9% on 10k blind holdout.",
  "hero.lead": "Chime is an offline-first neural Chinese Pinyin IME for macOS written in Rust and Swift 6. Chime v0.15.0 combines a lossless-slimmed 19.7 MB CHIMED02 lattice with first-principles Dual-Proposal Bayesian Route A beam search over a BI-18L mixed-precision MLX Qwen3 distilled from Qwen3-14B-Base (190.3 MB e8_l4, 36,864 full-CJK vocab), shrinking the installer from 257 MB to 192 MB while boosting blind holdout accuracy to 90.59% (80.93% on 10k).",
  "hero.dlPkg": "Download macOS Installer (.pkg)",
  "hero.dlZip": "Portable Chime.app (.zip)",
  "hero.dlZipSub": "Unzip into ~/Library/Input Methods",
  "m.base": "BI-18L Mixed-Precision MLX",
  "m.holdout": "Blind Holdout Acc@1",
  "m.speedup": "10k Blind Acc@1 (3× Faster)",
  "m.net": "Network & Telemetry",
  "demo.winTitle": "Interactive Candidate Bar & Full-Poem Tab Chaining",
  "demo.reset": "Reset Demo",
  "arch.eyebrow": "Route A Autoregressive Neural Beam Search + Multi-LoRA",
  "arch.title": "BI-18L Mixed-Precision (e8_l4), Dual-Proposal Bayesian Decoding & Zero-Jitter UX",
  "arch.desc": "Chime v0.15.0 pairs Block-Influence 18-layer wide-MLP Qwen3 distilled from Qwen3-14B-Base with mixed-precision MLX quantization (8-bit embed_tokens / lm_head + 4-bit layers) and exact Bayesian dual-proposal decoding — smaller, 3× faster, and more accurate than v0.14.",
  "arch.c1Title": "0 ms Fast Path + Async Dual-Proposal Route A Beam Search",
  "arch.c1Body": "Stage-1 lossless-slimmed CHIMED02 lattice (19.7 MB) renders instant candidates in < 0.8 ms on every keystroke, while background Route A beam search fuses lexicon proposals and neural hypotheses under one exact Bayesian objective.",
  "arch.c2Title": "BI-18L Qwen3-14B Exact-Tail Distill + e8_l4 Mixed Precision",
  "arch.c2Body": "Keeping embed_tokens / lm_head at 8-bit while quantizing the 18 wide-MLP layers to 4-bit recovers homophone ranking precision at just +18 MB and zero latency overhead — delivering 190.3 MB total model size and 3.01× faster 10k blind decoding.",
  "arch.c3Title": "Multi-LoRA Cartridges, Zero-Jitter UI & Silent Updates",
  "arch.c3Body": "Session-anchored candidate bar eliminates vertical jumping; localized surprisal-spike rescue recovers QWERTY slips in canonical sentences; and the Apple-notarized .pkg installs into ~/Library/Input Methods/Chime.app for silent passwordless updates.",
  "bench.eyebrow": "Strict Zero-Leakage Official Benchmarks",
  "bench.title": "Measured Accuracy Progression (v0.10.0 → v0.15.0)",
  "bench.desc": "Evaluated across five suites totaling 22,328 prompts — including two strictly blind holdouts (1,010 curated + 10,000 open-domain sentences) never inspected or used in training.",
  "packs.eyebrow": "Prebuilt .imepack Hub",
  "packs.title": "Official 18L Cartridges (Rank-8 LoRA + Zero-Copy Lexicon)",
  "packs.desc": "Install or update with 1 click directly inside Settings → Lexicon & AI (SHA-256 verified), or import any downloaded .imepack.zip locally / via CLI with zero restart.",
  "p1.sub": "Classical Poetry, Ci, Prose & Idioms Full Collection",
  "p1.body": "Includes Tang/Song poetry, Shijing, Chuci, Four Books, classic prose masterpieces, 30K idioms, and 14K allegorical sayings, paired with 231,593 stateful Tab verse continuation chains and an 18L Rank-8 classical Chinese LoRA (6.21 MB).",
  "p2.sub": "Systems Programming, AI & Cloud-Native Bilingual Pack",
  "p2.body": "Tuned for daily engineering workflows across Rust, Swift 6, Apple Silicon, MLX, Kubernetes, and distributed systems — preserving exact casing (macOS, GitHub, PostgreSQL) with an 18L Rank-8 developer LoRA (6.21 MB) auto-activated in terminals and IDEs.",
  "packs.dlClassics": "Download classics-poetry.imepack.zip",
  "packs.dlTech": "Download tech-developer.imepack.zip",
  "diy.eyebrow": "Bring Your Own Private LoRA",
  "diy.title": "Train a Strictly Private Personal Style Pack from Your Own Notes",
  "diy.desc": "Everyone writes with a distinct vocabulary. Chime ships with a local privacy scrubber (stripping API keys, emails, phone numbers, and private URLs) so you can train an 18L Rank-8 private_pack cartridge from your Markdown notes and mount it solely on your Mac."
};

const I18N_ZH = {};

let activeKey = "shuidiao";
let state = { committed: "", preedit: "", candidates: [], ghostIdx: -1, chain: [] };
let currentLang = "zh";

function loadPreset(key) {
  activeKey = key;
  const p = PRESETS[key];
  state = {
    committed: p.initialCommitted,
    preedit: p.preedit,
    candidates: p.candidates,
    ghostIdx: -1,
    chain: p.chain
  };
  document.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === key);
  });
  document.getElementById("ime-lora-badge").textContent = p.lora;
  renderDemo();
}

function commitFirstOrNextTab(viaTab) {
  if (state.preedit) {
    state.committed += state.candidates[0].text;
    state.preedit = "";
    state.candidates = [];
    state.ghostIdx = state.chain.length > 0 ? 0 : -1;
  } else if (viaTab && state.ghostIdx >= 0 && state.ghostIdx < state.chain.length) {
    state.committed += state.chain[state.ghostIdx];
    state.ghostIdx += 1;
    if (state.ghostIdx >= state.chain.length) {
      state.ghostIdx = -1;
    }
  }
  renderDemo();
}

function renderDemo() {
  document.getElementById("demo-committed").textContent = state.committed;
  document.getElementById("demo-preedit").textContent = state.preedit;

  const nextGhost =
    !state.preedit && state.ghostIdx >= 0 && state.ghostIdx < state.chain.length
      ? state.chain[state.ghostIdx]
      : "";
  document.getElementById("demo-ghost-inline").textContent = nextGhost ? `⇥ ${nextGhost}` : "";

  const candsEl = document.getElementById("ime-candidates");
  candsEl.innerHTML = "";
  if (state.preedit) {
    state.candidates.forEach((c, idx) => {
      const pill = document.createElement("div");
      pill.className = "cand-pill" + (idx === 0 ? " active" : "");
      pill.innerHTML = `<span class="cand-idx">${idx + 1}.</span><span>${c.text}</span>${
        c.tag ? `<span class="cand-tag">${c.tag}</span>` : ""
      }`;
      pill.addEventListener("click", () => {
        state.committed += c.text;
        state.preedit = "";
        state.candidates = [];
        state.ghostIdx = state.chain.length > 0 ? 0 : -1;
        renderDemo();
      });
      candsEl.appendChild(pill);
    });
    document.getElementById("ime-hint").textContent =
      currentLang === "zh" ? "点击候选或按 Space / Tab 上屏" : "Click candidate or press Space / Tab";
  } else if (nextGhost) {
    const ghostPill = document.createElement("div");
    ghostPill.className = "cand-pill active";
    ghostPill.innerHTML = `<span class="cand-idx">⇥ Tab</span><span>${nextGhost}</span><span class="cand-tag">诗词续写链 (${state.ghostIdx + 1}/${state.chain.length})</span>`;
    ghostPill.addEventListener("click", () => commitFirstOrNextTab(true));
    candsEl.appendChild(ghostPill);
    document.getElementById("ime-hint").textContent =
      currentLang === "zh" ? "继续点按「⇥ Tab」逐句背诵全篇" : "Keep pressing Tab to chain the full poem";
  } else {
    const doneSpan = document.createElement("span");
    doneSpan.style.fontSize = "0.84rem";
    doneSpan.style.color = "#a1a1aa";
    doneSpan.textContent =
      currentLang === "zh" ? "已完成上屏 · 点击上方预设标签切换场景" : "Committed · Pick another preset above";
    candsEl.appendChild(doneSpan);
    document.getElementById("ime-hint").textContent = "";
  }
}

document.querySelectorAll(".preset-chip").forEach((btn) => {
  btn.addEventListener("click", () => loadPreset(btn.dataset.preset));
});

document.getElementById("btn-tab").addEventListener("click", () => commitFirstOrNextTab(true));
document.getElementById("btn-space").addEventListener("click", () => commitFirstOrNextTab(false));
document.getElementById("btn-reset").addEventListener("click", () => loadPreset(activeKey));

document.addEventListener("keydown", (ev) => {
  if (ev.target && ["INPUT", "TEXTAREA"].includes(ev.target.tagName)) return;
  if (ev.key === "Tab" && (state.preedit || state.ghostIdx >= 0)) {
    ev.preventDefault();
    commitFirstOrNextTab(true);
  }
});

document.querySelectorAll(".copy-cmd-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const orig = btn.textContent;
    try {
      await navigator.clipboard.writeText(btn.dataset.cmd);
      btn.textContent = currentLang === "zh" ? "已复制 ✓" : "Copied ✓";
      setTimeout(() => {
        btn.textContent = orig;
      }, 1400);
    } catch (_) {}
  });
});

// Cache initial ZH strings and wire language toggle
document.querySelectorAll("[data-i18n]").forEach((el) => {
  I18N_ZH[el.dataset.i18n] = el.textContent;
});

document.getElementById("lang-toggle").addEventListener("click", (ev) => {
  currentLang = currentLang === "zh" ? "en" : "zh";
  ev.currentTarget.textContent = currentLang === "zh" ? "EN" : "中文";
  const dict = currentLang === "zh" ? I18N_ZH : I18N_EN;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    if (dict[el.dataset.i18n]) {
      el.textContent = dict[el.dataset.i18n];
    }
  });
  renderDemo();
});

// Load optional build-time SHA-256 manifest if present
fetch("releases-manifest.json")
  .then((r) => (r.ok ? r.json() : null))
  .then((manifest) => {
    if (!manifest) return;
    if (manifest.classics_sha256) {
      document.getElementById("sha-classics").textContent = manifest.classics_sha256.slice(0, 16) + "…";
      document.getElementById("sha-classics").title = manifest.classics_sha256;
    }
    if (manifest.tech_sha256) {
      document.getElementById("sha-tech").textContent = manifest.tech_sha256.slice(0, 16) + "…";
      document.getElementById("sha-tech").title = manifest.tech_sha256;
    }
  })
  .catch(() => {});

loadPreset("shuidiao");
