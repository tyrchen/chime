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

/// Populates all common Unicode CJK characters (including heteronym readings) so no character is
/// ever unreachable.
pub fn populate_standard_cjk_characters(lex: &mut Lexicon) -> usize {
    let mut added = 0usize;
    // Cover BMP CJK Unified Ideographs U+4E00..=U+9FA5 (20,902 characters) with realistic baseline
    // log-prob.
    for code in 0x4E00u32..=0x9FA5u32 {
        let Some(ch) = char::from_u32(code) else {
            continue;
        };
        let Some(multi) = ch.to_pinyin_multi() else {
            continue;
        };
        let surface = ch.to_string();
        for (idx, py) in multi.into_iter().take(3).enumerate() {
            // Unranked CJK fallback sits at -7.8 so real corpus-ranked characters always win!
            #[allow(clippy::cast_precision_loss)]
            let base_lp = -7.8 - (idx as f32) * 0.8;
            if lex.insert_pinyin_word(&surface, py.plain(), base_lp, false) {
                added += 1;
            }
        }
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
        if !(1..=6).contains(&char_len) {
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
        let scaled_lp = (-2.1 + (raw_ln + 4.5) * 0.31).clamp(-6.3, -2.2);
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
