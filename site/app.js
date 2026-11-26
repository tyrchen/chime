const PRESETS = {
  shuidiao: {
    lora: "LoRA: classics-poetry (28L Rank-8 · 1.0x)",
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
    lora: "LoRA: classics-poetry (28L Rank-8 · 防重复消歧)",
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
    lora: "LoRA: classics-poetry (28L Rank-8 · 乱序纠错 uq→qu)",
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
    lora: "LoRA: tech-developer (28L Rank-8 · 中/En 双模式英文纠错)",
    initialCommitted: "",
    preedit: "bang'wo'reivew'zhe'ge'pr",
    candidates: [
      { text: "帮我 review 这个 PR", tag: "自动纠错 reivew→review" },
      { text: "帮我 reivew 这个 PR", tag: "原词保留" }
    ],
    chain: []
  },
  singlegen: {
    lora: "Route A · Qwen3-28L 4-bit MLX (256.7 MB · 36K Full-CJK)",
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
  "hero.title": "Full 28-layer Qwen3 distilled from 14B on Apple Silicon — zero telemetry, 90% blind accuracy.",
  "hero.lead": "Chime is an offline-first neural Chinese Pinyin IME for macOS written in Rust and Swift 6. It combines a zero-copy mmap lattice fast path with Route A Pinyin-constrained autoregressive beam search over an intact 28-layer 4-bit MLX Qwen3-0.6B distilled from Qwen3-14B-Base (256.7 MB, 36,864 full-CJK vocab), fused 4× faster Metal inference, rock-solid Swift 6 UI, hot-swappable Multi-LoRA cartridges, and silent passwordless updates.",
  "hero.dlPkg": "Download macOS Installer (.pkg)",
  "hero.dlZip": "Portable Chime.app (.zip)",
  "hero.dlZipSub": "Unzip into ~/Library/Input Methods",
  "m.base": "28L 4-bit MLX Model",
  "m.holdout": "Blind Holdout Acc@1",
  "m.speedup": "Fused Metal Speedup",
  "m.net": "Network & Telemetry",
  "demo.winTitle": "Interactive Candidate Bar & Full-Poem Tab Chaining",
  "demo.reset": "Reset Demo",
  "arch.eyebrow": "Route A Autoregressive Neural Beam Search + Multi-LoRA",
  "arch.title": "Intact 28-Layer World Knowledge, 4× Faster Metal Inference & Zero-Jitter UX",
  "arch.desc": "Pure statistical dictionaries stall on long-range homophones and novel prose, while aggressive layer pruning erodes pretrained world knowledge. Chime v0.14.0 keeps all 28 layers of Qwen3-0.6B gently soft-KL distilled from Qwen3-14B-Base, fused for 4× faster Metal execution alongside hot-swappable .imepack cartridges.",
  "arch.c1Title": "0 ms Fast Path + Async Route A Neural Beam Search",
  "arch.c1Body": "Stage-1 zero-copy CHIMED02 lattice renders instant candidates in < 0.8 ms on every keystroke, while background Route A beam search incrementally advances an intact 28L Qwen3-0.6B over a deterministic TokenPinyinIndex Trie with zero UI blocking.",
  "arch.c2Title": "28L Qwen3-14B Soft-KL Base + 4× Fused Metal Kernels",
  "arch.c2Body": "Fused QKV and Gate+Up quantized projections cut 84 Metal GEMM launches per step, while single-barrier unpadded KV materialization eliminates cross-step lazy DAG blowup — delivering a 4.07× end-to-end speedup (p50 6.4 ms on contextual probes) with lossless 36,864 CJK vocabulary.",
  "arch.c3Title": "Multi-LoRA Cartridges, Zero-Jitter UI & Silent Updates",
  "arch.c3Body": "Session-anchored candidate bar eliminates vertical jumping; deviceIndependentFlagsMask dedup kills duplicate Shift toasts; and the v0.14.0 .pkg migrates Chime into ~/Library/Input Methods/Chime.app so all future updates run silently without password prompts or redundant macOS enable popups.",
  "bench.eyebrow": "Strict Zero-Leakage Official Benchmarks",
  "bench.title": "Measured Accuracy Progression (v0.10.0 → v0.14.0)",
  "bench.desc": "Evaluated across five suites totaling 22,328 prompts — including two strictly blind holdouts (1,010 curated + 10,000 open-domain sentences) never inspected or used in training.",
  "packs.eyebrow": "Prebuilt .imepack Hub",
  "packs.title": "Official 28L Cartridges (Rank-8 LoRA + Zero-Copy Lexicon)",
  "packs.desc": "Install or update with 1 click directly inside Settings → Lexicon & AI (SHA-256 verified), or import any downloaded .imepack.zip locally / via CLI with zero restart.",
  "p1.sub": "Classical Poetry, Ci, Prose & Idioms Full Collection",
  "p1.body": "Includes Tang/Song poetry, Shijing, Chuci, Four Books, classic prose masterpieces, 30K idioms, and 14K allegorical sayings, paired with 231,593 stateful Tab verse continuation chains and a 28L Rank-8 classical Chinese LoRA.",
  "p2.sub": "Systems Programming, AI & Cloud-Native Bilingual Pack",
  "p2.body": "Tuned for daily engineering workflows across Rust, Swift 6, Apple Silicon, MLX, Kubernetes, and distributed systems — preserving exact casing (macOS, GitHub, PostgreSQL) with a 28L Rank-8 developer LoRA auto-activated in terminals and IDEs.",
  "packs.dlClassics": "Download classics-poetry.imepack.zip",
  "packs.dlTech": "Download tech-developer.imepack.zip",
  "diy.eyebrow": "Bring Your Own Private LoRA",
  "diy.title": "Train a Strictly Private Personal Style Pack from Your Own Notes",
  "diy.desc": "Everyone writes with a distinct vocabulary. Chime ships with a local privacy scrubber (stripping API keys, emails, phone numbers, and private URLs) so you can train a 28L Rank-8 private_pack cartridge from your Markdown notes and mount it solely on your Mac."
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
