//! Permissive large-scale base lexicon generator (`MIT` `pinyin` + heteronym disambiguation
//! + optional `MIT` `jieba-rs` corpus frequencies + system English vocabulary).

use std::{collections::HashMap, path::Path};

use pinyin::{ToPinyin, ToPinyinMulti};

use crate::Lexicon;

/// Curated polyphonic (多音字) word readings where character-default Pinyin would be wrong!
const POLYPHONE_WORD_OVERRIDES: &[(&str, &str)] = &[
    ("银行", "yin hang"),
    ("行业", "hang ye"),
    ("同行", "tong hang"),
    ("行情", "hang qing"),
    ("行列", "hang lie"),
    ("排行榜", "pai hang bang"),
    ("一行", "yi hang"),
    ("换行", "huan hang"),
    ("命令行", "ming ling hang"),
    ("行为", "xing wei"),
    ("运行", "yun xing"),
    ("执行", "zhi xing"),
    ("进行", "jin xing"),
    ("并行", "bing xing"),
    ("流行", "liu xing"),
    ("旅行", "lv xing"),
    ("长大", "zhang da"),
    ("成长", "cheng zhang"),
    ("校长", "xiao zhang"),
    ("市长", "shi zhang"),
    ("部长", "bu zhang"),
    ("董事长", "dong shi zhang"),
    ("家长", "jia zhang"),
    ("增长", "zeng zhang"),
    ("擅长", "shan chang"),
    ("长度", "chang du"),
    ("长期", "chang qi"),
    ("长句", "chang ju"),
    ("音乐", "yin yue"),
    ("乐器", "yue qi"),
    ("乐队", "yue dui"),
    ("交响乐", "jiao xiang yue"),
    ("快乐", "kuai le"),
    ("娱乐", "yu le"),
    ("乐观", "le guan"),
    ("重庆", "chong qing"),
    ("重新", "chong xin"),
    ("重复", "chong fu"),
    ("重叠", "chong die"),
    ("重构", "chong gou"),
    ("重排", "chong pai"),
    ("重写", "chong xie"),
    ("重要", "zhong yao"),
    ("重点", "zhong dian"),
    ("重量", "zhong liang"),
    ("严重", "yan zhong"),
    ("尊重", "zun zhong"),
    ("权重", "quan zhong"),
    ("了解", "liao jie"),
    ("一目了然", "yi mu liao ran"),
    ("了不起", "liao bu qi"),
    ("睡觉", "shui jiao"),
    ("午觉", "wu jiao"),
    ("觉得", "jue de"),
    ("感觉", "gan jue"),
    ("视觉", "shi jue"),
    ("错觉", "cuo jue"),
    ("幻觉", "huan jue"),
    ("便宜", "pian yi"),
    ("占便宜", "zhan pian yi"),
    ("方便", "fang bian"),
    ("随便", "sui bian"),
    ("顺便", "shun bian"),
    ("即便", "ji bian"),
    ("着重", "zhuo zhong"),
    ("着手", "zhuo shou"),
    ("着陆", "zhuo lu"),
    ("执着", "zhi zhuo"),
    ("着急", "zhao ji"),
    ("睡不着", "shui bu zhao"),
    ("看着", "kan zhe"),
    ("接着", "jie zhe"),
    ("随着", "sui zhe"),
    ("意味着", "yi wei zhe"),
    ("还给", "huan gei"),
    ("归还", "gui huan"),
    ("偿还", "chang huan"),
    ("还原", "huan yuan"),
    ("还是", "hai shi"),
    ("还有", "hai you"),
    ("还好", "hai hao"),
    ("弹琴", "tan qin"),
    ("弹性", "tan xing"),
    ("弹窗", "tan chuang"),
    ("反弹", "fan tan"),
    ("子弹", "zi dan"),
    ("导弹", "dao dan"),
    ("炸弹", "zha dan"),
    ("厦门", "xia men"),
    ("大厦", "da sha"),
    ("西藏", "xi zang"),
    ("宝藏", "bao zang"),
    ("隐藏", "yin cang"),
    ("收藏", "shou cang"),
    ("存储", "cun chu"),
    ("准确率", "zhun que lv"),
    ("效率", "xiao lv"),
    ("概率", "gai lv"),
    ("频率", "pin lv"),
    ("帧率", "zhen lv"),
    ("心率", "xin lv"),
    ("率领", "shuai ling"),
    ("直率", "zhi shuai"),
    ("轻率", "qing shuai"),
    ("会计", "kuai ji"),
    ("会议", "hui yi"),
    ("学会", "xue hui"),
    ("参加", "can jia"),
    ("参数", "can shu"),
    ("参考", "can kao"),
    ("参与", "can yu"),
    ("人参", "ren shen"),
    ("参差", "cen ci"),
    ("参差不齐", "cen ci bu qi"),
    ("调整", "tiao zheng"),
    ("调试", "tiao shi"),
    ("微调", "wei tiao"),
    ("调用", "diao yong"),
    ("调查", "diao cha"),
    ("强调", "qiang diao"),
    ("色调", "se diao"),
    ("模样", "mu yang"),
    ("模具", "mu ju"),
    ("模型", "mo xing"),
    ("模块", "mo kuai"),
    ("模式", "mo shi"),
    ("模糊", "mo hu"),
    ("模仿", "mo fang"),
    ("规模", "gui mo"),
    ("给力", "gei li"),
    ("发给", "fa gei"),
    ("交给", "jiao gei"),
    ("供给", "gong ji"),
    ("自给自足", "zi ji zi zu"),
    ("处理", "chu li"),
    ("处于", "chu yu"),
    ("相处", "xiang chu"),
    ("到处", "dao chu"),
    ("好处", "hao chu"),
    ("深处", "shen chu"),
    ("数字", "shu zi"),
    ("数据", "shu ju"),
    ("数学", "shu xue"),
    ("函数", "han shu"),
    ("数不清", "shu bu qing"),
    ("数落", "shu luo"),
    ("朝阳", "chao yang"),
    ("朝向", "chao xiang"),
    ("朝代", "chao dai"),
    ("今朝", "jin zhao"),
    ("干活", "gan huo"),
    ("干净", "gan jing"),
    ("干燥", "gan zao"),
    ("骨干", "gu gan"),
    ("心脏", "xin zang"),
    ("内脏", "nei zang"),
    ("弄脏", "nong zang"),
    ("宿舍", "su she"),
    ("星宿", "xing xiu"),
    ("一宿", "yi xiu"),
    ("说服", "shuo fu"),
    ("说明", "shuo ming"),
    ("下载", "xia zai"),
    ("加载", "jia zai"),
    ("负载", "fu zai"),
    ("载体", "zai ti"),
    ("三年五载", "san nian wu zai"),
    ("省市", "sheng shi"),
    ("节省", "jie sheng"),
    ("省略", "sheng lve"),
    ("反省", "fan xing"),
    ("深省", "shen xing"),
    // `似` (`si` vs `shi`) and everyday high-frequency heteronym compounds
    ("似乎", "si hu"),
    ("相似", "xiang si"),
    ("类似", "lei si"),
    ("好似", "hao si"),
    ("近似", "jin si"),
    ("形似", "xing si"),
    ("神似", "shen si"),
    ("胜似", "sheng si"),
    ("恰似", "qia si"),
    ("似的", "shi de"),
    ("英文", "ying wen"),
    ("英语", "ying yu"),
    ("因为", "yin wei"),
    ("应该", "ying gai"),
    ("应当", "ying dang"),
    ("应用", "ying yong"),
    ("答应", "da ying"),
    ("反应", "fan ying"),
    ("反映", "fan ying"),
    ("角色", "jue se"),
    ("主角", "zhu jue"),
    ("配角", "pei jue"),
    ("角度", "jiao du"),
    ("角落", "jiao luo"),
    ("结束", "jie shu"),
    ("结果", "jie guo"),
    ("结合", "jie he"),
    ("结构", "jie gou"),
    ("结实", "jie shi"),
    ("漂亮", "piao liang"),
    ("漂浮", "piao fu"),
    ("钥匙", "yao shi"),
    ("汤匙", "tang chi"),
    ("薄弱", "bo ruo"),
    ("单薄", "dan bo"),
    ("薄荷", "bo he"),
    ("折腾", "zhe teng"),
    ("打折", "da zhe"),
    ("曝光", "bao guang"),
    ("暴露", "bao lu"),
    ("勉强", "mian qiang"),
    ("倔强", "jue jiang"),
    ("强大", "qiang da"),
    ("强迫", "qiang po"),
    ("屏幕", "ping mu"),
    ("屏蔽", "ping bi"),
    ("屏息", "bing xi"),
    ("提供", "ti gong"),
    ("提醒", "ti xing"),
    ("提防", "di fang"),
    ("埋怨", "man yuan"),
    ("埋伏", "mai fu"),
    ("山脉", "shan mai"),
    ("脉搏", "mai bo"),
    ("脉脉", "mo mo"),
    ("滑稽", "hua ji"),
    ("呼吁", "hu yu"),
    ("着迷", "zhao mi"),
    ("着火", "zhao huo"),
    ("着凉", "zhao liang"),
    ("大夫", "dai fu"),
    ("的确", "di que"),
    ("目的", "mu di"),
    ("标的", "biao di"),
];

/// Annotates a pure-CJK word into space-separated toneless Pinyin using overrides + `pinyin` crate.
#[must_use]
pub fn annotate_chinese_word<S: std::hash::BuildHasher>(
    word: &str,
    overrides: &HashMap<&str, &str, S>,
) -> Option<String> {
    if let Some(&ov) = overrides.get(word) {
        return Some(ov.to_string());
    }
    let mut syllables = Vec::with_capacity(word.chars().count());
    for ch in word.chars() {
        let py = ch.to_pinyin()?;
        syllables.push(py.plain());
    }
    if syllables.is_empty() {
        None
    } else {
        Some(syllables.join(" "))
    }
}

/// Genuine modern Mandarin dual-reading characters whose secondary reading is actively used in
/// contemporary Chinese (`长`, `行`, `重`, `乐`, `还`, `觉`, `似`, etc.). Prevents archaic Unihan
/// `to_pinyin_multi()` readings (`硬=geng`, `她=jie`, `听=yin`, `说=yue`) from polluting Top-5!
const MODERN_DAILY_HETERONYM_CHARS: &[(&str, &str, f32)] = &[
    ("长", "chang", -2.95),
    ("长", "zhang", -3.25),
    ("行", "xing", -2.90),
    ("行", "hang", -3.40),
    ("重", "zhong", -3.05),
    ("重", "chong", -3.45),
    ("乐", "le", -3.25),
    ("乐", "yue", -3.60),
    ("还", "hai", -2.65),
    ("还", "huan", -3.75),
    ("觉", "jue", -3.20),
    ("觉", "jiao", -3.65),
    ("着", "zhe", -2.85),
    ("着", "zhao", -3.55),
    ("着", "zhuo", -3.85),
    ("了", "le", -2.25),
    ("了", "liao", -3.70),
    ("的", "de", -2.05),
    ("的", "di", -4.15),
    ("地", "di", -3.05),
    ("地", "de", -3.55),
    ("得", "de", -2.85),
    ("得", "dei", -3.65),
    ("都", "dou", -2.65),
    ("都", "du", -3.75),
    ("大", "da", -2.55),
    ("似", "si", -3.55),
    ("似", "shi", -4.20),
    ("差", "cha", -3.35),
    ("差", "chai", -4.05),
    ("参", "can", -3.55),
    ("参", "shen", -4.25),
    ("数", "shu", -3.20),
    ("调", "tiao", -3.45),
    ("调", "diao", -3.55),
    ("模", "mo", -3.55),
    ("模", "mu", -4.20),
    ("奇", "qi", -3.55),
    ("奇", "ji", -4.35),
    ("弹", "tan", -3.75),
    ("弹", "dan", -3.80),
    ("省", "sheng", -3.45),
    ("省", "xing", -4.15),
    ("传", "chuan", -3.35),
    ("传", "zhuan", -4.05),
    ("率", "lv", -3.55),
    ("率", "shuai", -3.95),
    ("降", "jiang", -3.55),
    ("降", "xiang", -4.05),
    ("便", "bian", -3.35),
    ("便", "pian", -3.95),
    ("盛", "sheng", -3.75),
    ("盛", "cheng", -4.15),
    ("会", "hui", -2.65),
    ("会", "kuai", -4.25),
    ("给", "gei", -2.90),
    ("给", "ji", -4.15),
    ("藏", "cang", -3.65),
    ("藏", "zang", -3.95),
    ("朝", "chao", -3.55),
    ("朝", "zhao", -3.95),
    ("宿", "su", -3.85),
    ("宿", "xiu", -4.35),
    ("折", "zhe", -3.65),
    ("折", "she", -4.25),
    ("强", "qiang", -3.15),
    ("强", "jiang", -4.35),
    ("恶", "e", -3.65),
    ("恶", "wu", -4.25),
    ("单", "dan", -3.25),
    ("单", "shan", -4.35),
    ("解", "jie", -3.20),
    ("系", "xi", -3.25),
    ("壳", "ke", -3.95),
    ("壳", "qiao", -4.35),
    ("血", "xue", -3.55),
    ("血", "xie", -4.05),
    ("薄", "bo", -3.85),
    ("薄", "bao", -3.95),
    ("露", "lu", -3.65),
    ("露", "lou", -4.05),
    ("落", "luo", -3.35),
    ("落", "la", -4.25),
    ("称", "cheng", -3.40),
    ("没", "mei", -2.65),
    ("没", "mo", -3.95),
    ("跟", "gen", -2.95),
    ("根", "gen", -3.35),
];

/// Populates all common Unicode CJK characters (including heteronym readings) so no character is
/// ever unreachable.
pub fn populate_standard_cjk_characters(lex: &mut Lexicon) -> usize {
    let mut added = 0usize;
    // Cover BMP CJK Unified Ideographs U+4E00..=U+9FA5 (20,902 characters) with deep fallback
    // log-prob (-9.4) so obscure chars (`郠`, `绠`, `垜`) stay in the expanded single-char grid
    // and never pollute Top-5 candidates!
    for code in 0x4E00u32..=0x9FA5u32 {
        let Some(ch) = char::from_u32(code) else {
            continue;
        };
        let Some(multi) = ch.to_pinyin_multi() else {
            continue;
        };
        let surface = ch.to_string();
        for (idx, py) in multi.into_iter().take(3).enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let base_lp = -9.4 - (idx as f32) * 0.9;
            if lex.insert_pinyin_word(&surface, py.plain(), base_lp, false) {
                added += 1;
            }
        }
    }
    for &(ch_surf, py, lp) in MODERN_DAILY_HETERONYM_CHARS {
        let _ = lex.insert_pinyin_word(ch_surf, py, lp, false);
    }
    added
}

/// Builds a comprehensive, 100% permissive base [`Lexicon`] (~150,000+ entries) from:
/// 1. Curated seed & polyphone overrides
/// 2. All 20,902 CJK characters with heteronym readings (`pinyin`, MIT)
/// 3. Modern Chinese multi-character vocabulary & corpus frequencies (`jieba_dict_text`, MIT)
/// 4. Optional English wordlist (`/usr/share/dict/words`)
#[must_use]
pub fn build_comprehensive_lexicon(
    jieba_dict_text: &str,
    english_words_path: Option<&Path>,
    min_word_freq: u64,
) -> Lexicon {
    let mut lex = Lexicon::with_builtin_seed();
    let overrides: HashMap<&str, &str> = POLYPHONE_WORD_OVERRIDES.iter().copied().collect();

    // 1. Insert curated polyphone words with strong empirical weights.
    for &(word, py) in POLYPHONE_WORD_OVERRIDES {
        lex.insert_pinyin_word(word, py, -3.75, false);
    }

    // 2. Insert all 20,902 BMP CJK characters + heteronyms.
    let _ = populate_standard_cjk_characters(&mut lex);

    // 3. Parse Jieba corpus dictionary (`word freq [tag]`) and compute smoothed empirical log-prob.
    let mut rows: Vec<(&str, u64)> = Vec::new();
    let mut total_freq: f64 = 0.0;
    for raw_line in jieba_dict_text.lines() {
        let mut parts = raw_line.split_whitespace();
        let Some(word) = parts.next() else { continue };
        let Some(freq_str) = parts.next() else {
            continue;
        };
        let Ok(freq) = freq_str.parse::<u64>() else {
            continue;
        };
        let char_len = word.chars().count();
        if !(1..=12).contains(&char_len) {
            continue;
        }
        if char_len >= 2 && freq < min_word_freq {
            continue;
        }
        if !word.chars().all(|c| ('\u{4E00}'..='\u{9FA5}').contains(&c)) {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            total_freq += freq as f64;
        }
        rows.push((word, freq));
    }

    let denom = total_freq.max(1.0).ln();
    for (word, freq) in rows {
        let Some(py) = annotate_chinese_word(word, &overrides) else {
            continue;
        };
        // Empirical log(freq / total) normally spans [-17.5, -4.5]; compress smoothly into [-6.2,
        // -2.3] to match our lattice dynamic range!
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let raw_ln = ((freq.max(1) as f64).ln() - denom) as f32;
        let scaled_lp = (-2.35 + (raw_ln + 4.5) * 0.32).clamp(-6.6, -2.45);
        lex.insert_pinyin_word(word, &py, scaled_lp, false);
    }

    // 4. Optional English dictionary words (common lowercase words length 4..=14).
    if let Some(eng_path) = english_words_path
        && let Ok(eng_text) = std::fs::read_to_string(eng_path)
    {
        for line in eng_text.lines() {
            let w = line.trim();
            if (4..=14).contains(&w.len())
                && w.bytes().all(|b| b.is_ascii_lowercase())
                && lex.lookup_english(&w.to_ascii_lowercase()).is_none()
            {
                lex.insert_english_term(w, -5.4, false);
            }
        }
    }

    lex
}
