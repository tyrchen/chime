const PRESETS = {
  shuidiao: {
    lora: "LoRA: classics-poetry (1.0x)",
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
    lora: "LoRA: classics-poetry (1.0x · 防重复消歧)",
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
    lora: "LoRA: classics-poetry (1.0x · 乱序纠错 uq→qu)",
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
    lora: "LoRA: tech-developer (1.0x in Ghostty / VS Code)",
    initialCommitted: "",
    preedit: "wo'shi'yong'macos'he'github'kaifa",
    candidates: [
      { text: "我使用 macOS 和 GitHub 开发", tag: "免切换混输" },
      { text: "我试用 macOS 和 GitHub 开发", tag: "" }
    ],
    chain: []
  },
  singlegen: {
    lora: "Base Qwen3-6L (4-bit MLX · 64.2 MB)",
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
  "nav.arch": "Base + Multi-LoRA",
  "nav.packs": "LoRA Pack Hub",
  "nav.diy": "Private LoRA",
  "hero.eyebrow": "macOS 15+ · Apple Silicon · Apple Notarized · 100% Offline · MIT / Apache-2.0",
  "hero.title": "A 64 MB local neural IME on Unified Memory, with hot-swappable LoRA cartridges.",
  "hero.lead": "Chime is an offline-first Chinese Pinyin input method for Apple Silicon. Written in Rust and Swift 6, it pairs a <0.8 ms zero-copy mmap lattice decoder with a 6-layer 4-bit MLX Qwen3 reranker and 2 MB plug-and-play Multi-LoRA .imepack cartridges.",
  "hero.dlPkg": "Download macOS Installer (.pkg)",
  "hero.dlZip": "Portable Chime.app (.zip)",
  "hero.dlZipSub": "Drag into ~/Library/Input Methods",
  "m.base": "4-bit MLX Base Model",
  "m.lora": "Per Rank-8 LoRA Pack",
  "m.lat": "End-to-End p50 Latency",
  "m.net": "Network & Telemetry",
  "demo.winTitle": "Interactive Candidate Bar & Full-Poem Tab Chaining",
  "demo.reset": "Reset Demo",
  "arch.eyebrow": "Base Model + Multi-LoRA Sidecar",
  "arch.title": "Why One Frozen Base Model + Pluggable .imepack Cartridges?",
  "arch.desc": "Pure statistical dictionaries miss contextual homophones, while a LoRA alone cannot recall verses missing from Stage 1. Each .imepack bundles zero-copy CHIMED02 recall + continuation chains alongside a 2 MB Rank-8 Metal LoRA.",
  "packs.eyebrow": "Prebuilt .imepack Hub",
  "packs.title": "Official Prebuilt Cartridges (LoRA + Zero-Copy Lexicon)",
  "packs.desc": "Import any downloaded .imepack.zip directly from Menu Bar → Chime → Import Dictionary or .imepack, or via CLI. Hot-mounts with zero restart.",
  "packs.dlClassics": "Download classics-poetry.imepack.zip",
  "packs.dlTech": "Download tech-developer.imepack.zip",
  "diy.eyebrow": "Bring Your Own Private LoRA",
  "diy.title": "Train a Strictly Private Personal Style Pack from Your Own Notes"
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
