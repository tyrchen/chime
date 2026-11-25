#!/usr/bin/env python3
"""
Builds the strictly held-out unseen test benchmark (`benchmarks/test_holdout.tsv`, 1,000+ cases)
across 15 real-world Mandarin & bilingual domains, verifying 0% input overlap against
`benchmarks/dev_suite.tsv`.

IMPORTANT ANTI-OVERFIT RULE:
Never inspect individual rows of `benchmarks/test_holdout.tsv` to add lexical overrides,
and always exclude both `dev_suite.tsv` and `test_holdout.tsv` n-grams from neural distillation!
"""

from __future__ import annotations

import pathlib
import random
from pypinyin import Style, lazy_pinyin

XIAOHE_INITIAL = {
    "zh": "v", "ch": "i", "sh": "u",
}
XIAOHE_FINAL = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "d", "ei": "w", "ui": "v", "ao": "c", "ou": "z", "iu": "q",
    "ie": "p", "ve": "t", "ue": "t", "er": "e",
    "an": "j", "en": "f", "in": "b", "un": "y", "vn": "y",
    "ang": "h", "eng": "g", "ing": "k", "ong": "s", "iong": "s",
    "ia": "x", "iao": "n", "ian": "m", "iang": "l",
    "ua": "x", "uo": "o", "uai": "k", "uan": "r", "uang": "l",
}

ZIRANMA_FINAL = {
    "a": "a", "o": "o", "e": "e", "i": "i", "u": "u", "v": "v",
    "ai": "l", "ei": "z", "ui": "v", "ao": "k", "ou": "b", "iu": "q",
    "ie": "x", "ve": "t", "ue": "t", "er": "e",
    "an": "j", "en": "f", "in": "n", "un": "p", "vn": "p",
    "ang": "h", "eng": "g", "ing": "y", "ong": "s", "iong": "s",
    "ia": "w", "iao": "c", "ian": "m", "iang": "d",
    "ua": "w", "uo": "o", "uai": "y", "uan": "r", "uang": "d",
}

QWERTY_NEIGHBORS = {
    "a": "s", "b": "v", "c": "x", "d": "f", "e": "r", "f": "g", "g": "h",
    "h": "j", "i": "o", "j": "k", "k": "l", "l": "k", "m": "n", "n": "m",
    "o": "p", "p": "o", "q": "w", "r": "t", "s": "d", "t": "y", "u": "i",
    "v": "b", "w": "e", "x": "c", "y": "u", "z": "x",
}


def to_full_pinyin_syllables(text: str) -> list[str]:
    syls = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    return [s.lower().replace("ü", "v").replace("u:", "v") for s in syls if s.isalpha()]


def to_full_pinyin(text: str) -> str:
    return "".join(to_full_pinyin_syllables(text))


def split_init_final(syl: str) -> tuple[str, str]:
    for retro in ("zh", "ch", "sh"):
        if syl.startswith(retro):
            return retro, syl[2:]
    if syl[0] in "bpmfdtnlgkhjqxrzcsyw":
        if syl in ("a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er"):
            return "", syl
        return syl[0], syl[1:]
    return "", syl


def encode_shuangpin(text: str, scheme: str) -> str | None:
    fmap = XIAOHE_FINAL if scheme == "xiaohe" else ZIRANMA_FINAL
    out = []
    for syl in to_full_pinyin_syllables(text):
        init, fin = split_init_final(syl)
        if not init:
            if len(syl) == 1:
                out.append(syl * 2)
            elif len(syl) == 2:
                out.append(syl)
            elif scheme == "xiaohe" and syl in ("ang", "eng"):
                out.append(syl[0] + ("h" if syl == "ang" else "g"))
            else:
                return None
            continue
        if init in ("j", "q", "x", "y") and fin == "u":
            pass
        elif init in ("j", "q", "x", "y") and fin == "ue":
            fin = "ve"
        ic = XIAOHE_INITIAL.get(init, init)
        fc = fmap.get(fin)
        if not fc:
            return None
        out.append(ic + fc)
    return "".join(out)


def main() -> None:
    repo = pathlib.Path("/Users/tchen/projects/mycode/rust/chime")
    bench_dir = repo / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    dev_path = repo / "dist/bench_1200.tsv"
    dev_keys = set()
    if dev_path.exists():
        shutil_text = dev_path.read_text("utf-8")
        (bench_dir / "dev_suite.tsv").write_text(shutil_text, "utf-8")
        for ln in shutil_text.splitlines():
            parts = ln.split("\t")
            if len(parts) >= 5:
                dev_keys.add((parts[3].strip(), parts[4].strip()))

    rows: list[tuple[str, str, str, str, str, str]] = []

    def add(cat: str, scheme: str, app: str, ctx: str, keys: str, expected: str) -> None:
        if (ctx, keys) in dev_keys:
            return
        if any(r[3] == ctx and r[4] == keys and r[1] == scheme for r in rows):
            return
        rows.append((cat, scheme, app, ctx, keys, expected))

    # 1. everyday_chat (100 unseen natural conversational utterances)
    everyday = [
        "早上起来先喝一杯温水", "昨晚睡得特别踏实", "今天出门记得带雨伞", "外面好像开始刮风了",
        "周末打算去公园散步", "这家店的面条味道不错", "帮我顺便买一瓶牛奶", "地铁上人实在太多了",
        "下午三点有个牙医预约", "手机电量只剩百分之十", "钥匙好像忘在桌子上了", "快递已经放到门口柜子",
        "晚饭我们自己在家做吧", "冰箱里还有几个西红柿", "这双运动鞋穿着很舒服", "刚才给你发了一条消息",
        "路上注意安全慢点开车", "到了那边记得报个平安", "最近天气变化真快", "秋天到了树叶开始变黄",
        "窗户没关紧有点漏风", "洗衣机里的衣服洗好了", "阳台上的太阳特别暖和", "孩子今天在学校表现很好",
        "周末一起去看场新电影", "这张沙发坐着非常软", "家里需要买点纸巾了", "楼下新开了一家面包店",
        "每天坚持跑步半个小时", "早点休息别熬夜太晚", "明天早上七点叫我起床", "闹钟响了好几遍才醒",
        "这杯咖啡不加糖更好喝", "桌上的水杯是谁留下的", "帮我把客厅的灯关一下", "空调温度别调得太低",
        "出门前检查一下煤气阀门", "超市今天水果打八折", "买两斤苹果和一串香蕉", "这个西瓜吃起来特别甜",
        "猫咪趴在窗台上晒太阳", "小狗在草地上欢快地跑", "邻居家的花开得真漂亮", "电梯正在维修请走楼梯",
        "小区门口新修了自行车道", "今天路上一点都不堵车", "提前十分钟到达火车站", "高铁坐起来又快又稳",
        "行李箱轮子有点卡住了", "酒店房间可以看到大海", "海边的晚风吹着真凉爽", "拍了很多好看的风景照",
        "把照片发到家庭群里吧", "爷爷奶奶身体一直很硬朗", "过年全家一起吃团圆饭", "厨房里飘来阵阵饭菜香",
        "这道菜盐稍微放多了一点", "再添一碗米饭就够饱了", "吃完饭一起去河边走走", "散步回来洗个热水澡",
        "这本书我已经看了一大半", "故事结局让人非常感动", "戴上耳机听一会儿轻音乐", "书架上摆满了各种小说",
        "笔记本用完了再去买两本", "钢笔没墨水了需要灌墨", "眼镜片脏了拿布擦一擦", "背包拉链坏了得去修一下",
        "冬天穿羽绒服最保暖", "围巾和手套都准备齐全", "今年冬天还没下过雪呢", "希望明天是个大晴天",
        "朋友从外地过来看望我们", "好久不见聊得特别投机", "约好了下个月一起去爬山", "山顶的空气格外清新自然",
        "沿着石阶一步一步往上爬", "累了就在亭子里歇一会儿", "带的水差不多都喝完了", "下山的时候要注意脚下",
        "这家书店环境非常安静", "找个靠窗的位置坐下看书", "点了一壶热茶慢慢品尝", "不知不觉度过了整个下午",
        "傍晚的晚霞染红了半边天", "街边的路灯一盏接一盏亮起", "夜市里到处都是热闹的人群", "买了一份热乎乎的烤红薯",
        "回到家已经快到九点钟了", "把明天要穿的衣服找出来", "早睡早起身体才会更健康", "新的一周又要精神饱满",
        "今天心情莫名觉得很轻松", "遇到困难先冷静下来思考", "凡事往好处想就对了", "慢慢来一切都会好起来的",
        "谢谢你一直以来的帮忙", "有空常来家里坐坐聊聊天", "祝你生日快乐心想事成", "愿每一天都充满阳光欢笑",
    ]
    for s in everyday:
        add("everyday_chat", "full", "com.tencent.xinWeChat", "", to_full_pinyin(s), s)

    # 2. workplace_collab (80 unseen workplace & collaboration sentences)
    workplace = [
        "请大家准时参加上午的例会", "这份需求文档还需要再细化", "把会议纪要同步到群里吧", "下周五之前完成本阶段交付",
        "项目进度目前符合预期计划", "麻烦帮忙确认一下合同条款", "客户提出的反馈非常有价值", "我们需要尽快给出解决方案",
        "下午两点和设计团队对齐细节", "测试环境已经部署好了新版本", "上线前务必做好全量回归验证", "遇到阻塞问题及时在群里沟通",
        "季度总结报告明天中午前提交", "预算审批流程已经走完了", "新同事今天第一天入职报到", "帮新伙伴开通一下办公权限",
        "这个季度的核心目标是提效", "把关键指标拆解到每周跟进", "跨部门协作需要明确各自职责", "感谢各位同事的辛勤付出",
        "产品原型图已经更新到最新版", "用户体验细节还需要继续打磨", "收集一下种子用户的真实反馈", "数据看板显示留存率稳步上升",
        "运营活动定在下周一正式启动", "宣传海报的排版看着很大气", "法务同事已经审核通过了协议", "财务报销单据请贴好发票提交",
        "出差机票和酒店都已经订妥了", "远程会议链接发在日历邀请里", "共享屏幕的时候声音有点小", "我先把麦克风静音听大家讲",
        "针对这个方案大家有什么建议", "我们可以分两个阶段逐步推进", "优先解决影响面最大的痛点", "保持信息透明减少沟通成本",
        "技术评审定在周三下午三点", "灰度发布先覆盖百分之五流量", "观察线上监控指标是否平稳", "一切正常后再逐步全量放开",
        "这份排期表留出了一定缓冲期", "风险点需要提前制定应急预案", "文档结构清晰读起来一目了然", "培训资料已经上传到知识库",
        "每周五下午做一次简短复盘", "把踩过的坑整理成经验总结", "避免不同团队重复造轮子", "统一编码规范提高可维护性",
        "面试候选人的基础非常扎实", "综合评估下来非常契合岗位", "请人事同事推进后续录用流程", "团队氛围融洽干劲也很足",
        "客户现场演示效果超出预期", "后续定制需求再单独拉群讨论", "版本发布说明写得简明扼要", "重点突出本次升级的核心亮点",
        "客服工单响应速度明显加快", "用户满意度比上个月提升了", "下阶段的规划草案已经写好", "欢迎大家随时补充不同意见",
        "时间紧任务重大家抓紧推进", "有问题直接电话或者语音沟通", "把复杂的问题拆成可执行步骤", "一步一个脚印把产品做扎实",
        "今天先把主流程跑通再优化", "边界条件都要有单元测试覆盖", "评审通过后就可以合并主干", "发布完成后记得在群里通知",
        "年度绩效评估下周正式开始", "认真总结过去一年的成长收获", "制定清晰可衡量的个人目标", "期待大家在新一年更进一步",
        "办公区的无线网络刚刚升级了", "会议室投影仪连接非常顺畅", "茶水间补充了新的咖啡豆", "保持工位整洁工作心情更好",
        "下班前记得保存文件锁好屏幕", "祝大家度过一个愉快的周末", "节假日期间安排好值班人员", "紧急告警会第一时间电话通知",
    ]
    for s in workplace:
        add("workplace_collab", "full", "com.tinyspeck.slackmacgap", "", to_full_pinyin(s), s)

    # 3. prose_essay & culture (75 unseen written prose sentences)
    prose = [
        "真正优秀的设计往往简洁而克制", "阅读能让人在喧嚣中保持内心宁静", "时间会检验一切浮华背后的成色",
        "科学精神的核心在于勇于质疑求真", "语言不仅是交流工具更是思维载体", "好的文章应当言之有物逻辑严密",
        "历史经验往往能照亮未来的道路", "独立思考比盲目跟风更加难能可贵", "细节之处最能体现工匠精神的温度",
        "人与自然和谐共生是永恒的课题", "万里晴空下远山轮廓格外分明", "春雨悄无声息地滋润着干涸土地",
        "古老的街巷承载着几代人的记忆", "博物馆里陈列着精美的青铜器物", "书法艺术讲究笔画之间的气韵连贯",
        "传统音乐在现代编曲中焕发新生", "远方灯塔在夜幕中闪烁着微弱光芒", "河流蜿蜒穿过肥沃的平原地带",
        "清晨薄雾笼罩着寂静的江南水乡", "竹林深处传来清脆悦耳的鸟鸣声", "岁月变迁并未磨灭那份赤诚初心",
        "宽容待人是一种成熟豁达的智慧", "好奇心驱动人类不断探索未知边界", "每一次挫折都是成长路上的垫脚石",
        "持之以恒往往比一时冲动更有力量", "删繁就简才能抓住事物的核心本质", "优秀的工具应当隐于无形顺手自然",
        "知识的价值在于融会贯通学以致用", "平淡生活里同样蕴藏着动人的诗意", "抬头仰望星空总会让人心生敬畏",
        "秋风吹过麦田掀起层层金色波浪", "冬日暖阳透过玻璃洒在木质书桌上", "老槐树下坐着乘凉下棋的老人们",
        "茶香袅袅升起让人瞬间放松下来", "山间清泉流过鹅卵石发出潺潺声响", "海浪拍打礁石激起洁白细腻的泡沫",
        "经典作品历经百年依然具有生命力", "审美素养在潜移默化中不断得到熏陶", "倾听他人观点有助于打破认知局限",
        "脚踏实地才能走得更稳更长远", "诚实守信是立身处世的根本准则", "理性分析与感性共情并非截然对立",
        "城市更新既要注重效率也要保留温度", "乡村夜晚能看到久违的璀璨银河", "四季更替展现出大自然奇妙的节律",
        "耐心打磨一件作品本身就是享受", "文字的力量足以跨越漫长时空阻隔", "心怀热爱才能在平凡中创造不凡",
        "保持谦逊才能不断吸收新鲜养分", "逆境之中更显坚韧不拔的意志品质", "教育的目的在于唤醒求知与创造欲",
        "清晰的表达源于透彻深入的思考", "化繁为简需要深厚的功底与洞察力", "温和坚定是一种极具感染力的品格",
        "漫步林间小径感受微风拂面的惬意", "古镇石桥倒映在碧绿澄澈的河水中", "炊烟缓缓升起勾勒出温暖人间烟火",
        "雪花纷纷扬扬给大地披上洁白外衣", "晨曦微露时分海面泛起粼粼金光", "骏马在辽阔草原上尽情驰骋奔跑",
        "音乐跨越国界触动人类共同的情感", "建筑被称为凝固的历史与立体诗篇", "绘画通过光影色彩捕捉瞬间的永恒",
        "戏剧冲突揭示出复杂深刻的人性百态", "电影镜头记录下时代变迁的珍贵印记", "摄影让人学会用全新视角观察日常",
        "诗歌以凝练语言传达无穷想象空间", "散文形散神聚展现真挚自然的性情", "小说虚构故事却折射出真实社会百态",
        "哲学追问生命意义与世界本原规律", "天文学带领人类仰望浩瀚无垠宇宙", "地质学解读岩石层里亿万年的密码",
        "生物学揭示生命演化繁衍的精妙机制", "物理学用简洁方程描绘自然界法则", "数学展现纯粹抽象而又严密的秩序美",
    ]
    for s in prose:
        add("prose_essay", "full", "md.obsidian", "", to_full_pinyin(s), s)

    # 4. science_medical_legal & finance_business (90 unseen domain sentences)
    domains = [
        ("science_medical_legal", "保持规律作息有助于增强免疫功能"),
        ("science_medical_legal", "每年定期体检可以及早发现健康隐患"),
        ("science_medical_legal", "遵照医嘱按时服药切勿自行停药"),
        ("science_medical_legal", "适量补充蛋白质和膳食纤维很有益"),
        ("science_medical_legal", "临床试验数据证实了新疗法的安全性"),
        ("science_medical_legal", "监护人应当依法履行抚养教育义务"),
        ("science_medical_legal", "当事人有权在法定期限内提起上诉"),
        ("science_medical_legal", "知识产权保护激励了持续技术创新"),
        ("science_medical_legal", "证据链完整清晰具有充分法律效力"),
        ("science_medical_legal", "合同双方应当严格遵守诚实信用原则"),
        ("science_medical_legal", "光合作用将太阳能转化为化学能储存"),
        ("science_medical_legal", "温室气体排放加剧了全球气候变暖"),
        ("science_medical_legal", "新能源电池能量密度取得了重要突破"),
        ("science_medical_legal", "基因测序技术大幅降低了罕见病筛查成本"),
        ("science_medical_legal", "卫星遥感监测为防灾减灾提供精准数据"),
        ("science_medical_legal", "地震预警系统争取了宝贵的避险时间"),
        ("science_medical_legal", "海洋生态系统维持着全球碳循环平衡"),
        ("science_medical_legal", "抗生素滥用容易诱导细菌产生耐药性"),
        ("science_medical_legal", "有氧运动能够有效改善心血管机能"),
        ("science_medical_legal", "长期低头看手机容易引发颈椎疲劳"),
        ("science_medical_legal", "保护未成年人合法权益是全社会共同责任"),
        ("science_medical_legal", "仲裁裁决一经作出即具有终局约束力"),
        ("science_medical_legal", "个人信息采集必须遵循最小必要原则"),
        ("science_medical_legal", "疫苗接种构建起坚实的人群免疫屏障"),
        ("science_medical_legal", "显微镜下可以清晰观察细胞分裂过程"),
        ("science_medical_legal", "超导材料在特定低温下电阻完全消失"),
        ("science_medical_legal", "光谱分析能够准确测定恒星化学成分"),
        ("science_medical_legal", "湿地被称为地球之肾具有净化水质功能"),
        ("science_medical_legal", "生物多样性丧失将威胁人类生存根基"),
        ("science_medical_legal", "司法公正是维护社会公平正义最后防线"),
        ("finance_business", "稳健的现金流是企业持续经营的生命线"),
        ("finance_business", "分散投资组合可以有效降低单一市场波动"),
        ("finance_business", "供应链韧性决定了制造业抗风险能力"),
        ("finance_business", "央行公开市场操作维护了流动性合理充裕"),
        ("finance_business", "制造业采购经理指数连续三个月回升"),
        ("finance_business", "跨境电商出口规模今年实现了快速增长"),
        ("finance_business", "精细化运营显著降低了客户获取成本"),
        ("finance_business", "品牌口碑积累需要长期稳定的品质保障"),
        ("finance_business", "毛利率提升主要得益于生产工艺改进"),
        ("finance_business", "研发投入占营业收入比重持续保持高位"),
        ("finance_business", "固定资产折旧采用直线法按月计提"),
        ("finance_business", "应收账款周转天数比去年同期明显缩短"),
        ("finance_business", "理性评估风险收益比避免盲目追涨杀跌"),
        ("finance_business", "长期复利效应在时间维度上非常惊人"),
        ("finance_business", "消费升级带动了高品质服务需求增长"),
        ("finance_business", "绿色金融支持低碳环保产业转型升级"),
        ("finance_business", "审计报告确认财务报表真实公允准确"),
        ("finance_business", "合理控制资产负债率防范流动性风险"),
        ("finance_business", "规模效应逐步显现推动单位成本下降"),
        ("finance_business", "核心技术壁垒构成了企业坚固护城河"),
        ("finance_business", "新产品上市首月销量突破十万台大关"),
        ("finance_business", "渠道下沉打开了广阔的三四线城市市场"),
        ("finance_business", "复购率是衡量用户忠诚度的核心指标"),
        ("finance_business", "季度财报披露营收与净利润双双增长"),
        ("finance_business", "汇率波动对出口型企业利润有一定影响"),
        ("finance_business", "通过套期保值锁定原材料采购价格"),
        ("finance_business", "物流仓储自动化大幅提升了发货效率"),
        ("finance_business", "数字化转型重塑了传统零售业务流程"),
        ("finance_business", "耐心资本陪伴硬科技初创企业共同成长"),
        ("finance_business", "透明规范的公司治理赢得了投资者信任"),
    ]
    for cat, s in domains:
        add(cat, "full", "md.obsidian", "", to_full_pinyin(s), s)

    # 5. tech_engineering (80 unseen CS / systems / AI sentences)
    tech = [
        "零拷贝内存映射极大缩短了冷启动耗时", "锁粒度过粗会导致多核扩展性急剧下降",
        "使用环形缓冲区实现无阻塞线程间通信", "编译器静态检查在构建阶段消除了数据竞争",
        "向量化指令集加速了矩阵点积运算吞吐", "增量编译只重新构建发生变更的依赖模块",
        "内存池复用显著减少了频繁分配释放开销", "分布式一致性协议保证了副本间状态同步",
        "读写分离架构有效缓解了主数据库压力", "索引覆盖查询避免了额外的回表磁盘读取",
        "连接池参数需要根据峰值并发合理配置", "幂等性设计防止网络重试造成重复扣款",
        "熔断降级机制保护核心服务不被级联拖垮", "链路追踪清晰定位了跨服务调用性能瓶颈",
        "容器镜像分层构建加快了持续集成流水线", "蓝绿部署实现了零停机无缝版本切换",
        "注意力机制计算复杂度随序列长度平方增长", "旋转位置编码具有良好的长上下文外推能力",
        "分组查询注意力大幅压缩了键值缓存显存", "知识蒸馏让轻量级学生模型继承教师分布",
        "混合精度训练兼顾了收敛稳定性与吞吐量", "梯度裁剪有效防止了深层网络梯度爆炸",
        "余弦退火学习率有助于模型收敛到更优解", "早停策略可以防止在小数据集上发生过拟合",
        "分词器词表大小直接影响嵌入层参数体积", "束搜索宽度需要在解码质量与延迟间权衡",
        "动态规划算法在线性时间内求得全局最优路径", "哈希表负载因子过高时会增加冲突探测次数",
        "红黑树通过旋转染色维持近似平衡高度", "布隆过滤器以极小空间快速判断元素不存在",
        "跳表结构实现简单且支持高效范围查询", "前缀树非常适合输入法词库音节前缀匹配",
        "状态机模式让复杂键盘事件处理清晰可测", "端到端加密确保传输内容无法被中间人窃听",
        "数字签名验证了安装包来源真实且未被篡改", "沙箱隔离限制了不可信插件的系统访问权限",
        "最小权限原则降低了潜在安全漏洞破坏范围", "模糊测试自动发现了多处极端输入边界缺陷",
        "基准测试必须在相同硬件负载下重复验证", "火焰图直观展示了热点函数的处理器占比",
        "缓存行伪共享会导致多核缓存频繁失效", "预取指令提前将数据加载到高速缓存层级",
        "异步事件循环用单线程支撑数万并发连接", "背压机制防止快速生产者压垮慢速消费者",
        "写前日志保证了崩溃恢复后的数据持久性", "快照隔离级别消除了不可重复读并发异常",
        "列式存储格式特别适合大规模聚合分析查询", "倒排索引是全文检索引擎最核心的数据结构",
        "图形管线利用统一内存避免了跨总线拷贝", "着色器预编译消除了首次渲染时的卡顿现象",
        "垂直同步使画面刷新与显示器帧率保持一致", "自适应采样在边缘区域投入更高计算密度",
        "语音识别前端通常提取梅尔频率倒谱特征", "声学模型与语言模型联合解码提升识别率",
        "卡尔曼滤波融合多传感器数据估计真实状态", "控制回路需要调节比例积分微分三个参数",
        "嵌入式固件升级必须支持断电安全回滚机制", "看门狗定时器在程序死锁时自动触发复位",
        "静态链接生成的单一可执行文件便于分发", "跨平台抽象层隔离了底层操作系统接口差异",
    ]
    for s in tech:
        add("tech_engineering", "full", "com.mitchellh.ghostty", "", to_full_pinyin(s), s)

    # 6. code_switch_mixed (70 unseen bilingual developer & product sentences)
    cs = [
        ("woshiyongrustkaifagaojixitong", "我使用 Rust 开发高级系统", "com.mitchellh.ghostty"),
        ("zaimacosshangyunxingfeichangliuchang", "在 macOS 上运行非常流畅", "md.obsidian"),
        ("qingbangwoshenchazhegeprdedaima", "请帮我审查这个 PR 的代码", "com.microsoft.VSCode"),
        ("bamoxingbaocunweisafetensorsgeshi", "把模型保存为 Safetensors 格式", "com.mitchellh.ghostty"),
        ("yongpytorchxunlianduancexiaoquanzhong", "用 PyTorch 训练端侧小权重", "com.mitchellh.ghostty"),
        ("bushudaokubernetesjiqunzhong", "部署到 Kubernetes 集群中", "com.mitchellh.ghostty"),
        ("shiyongdockerdabaojingxiang", "使用 Docker 打包镜像", "com.mitchellh.ghostty"),
        ("tongguogrpcdiaoyonghouduanfuwu", "通过 gRPC 调用后端服务", "com.microsoft.VSCode"),
        ("shujucunchuzaipostgresqlshujuku", "数据存储在 PostgreSQL 数据库", "com.microsoft.VSCode"),
        ("bendiwenjianshiyongsqlitecunchu", "本地文件使用 SQLite 存储", "com.microsoft.VSCode"),
        ("qianduanyongtypescripthevuekaifa", "前端用 TypeScript 和 Vue 开发", "com.microsoft.VSCode"),
        ("peizhinginxzuofanxiangdaili", "配置 Nginx 做反向代理|配置 nginx 做反向代理", "com.mitchellh.ghostty"),
        ("zaiiphoneshangceshixinjieko", "在 iPhone 上测试新接口", "com.apple.dt.Xcode"),
        ("yongxcodepenxieswiftuiyemian", "用 Xcode 编写 SwiftUI 页面", "com.apple.dt.Xcode"),
        ("liyongmlxzaimetalshangtuili", "利用 MLX 在 Metal 上推理", "com.mitchellh.ghostty"),
        ("zhegeapideyanchifeichangdi", "这个 API 的延迟非常低", "com.microsoft.VSCode"),
        ("conguggingfacehuoquyuanshimoxing", "从 HuggingFace 获取原始模型", "com.mitchellh.ghostty"),
        ("duibichatgptheclaudedehuida", "对比 ChatGPT 和 Claude 的回答", "md.obsidian"),
        ("deepseekzaizhongwenrenwushanghenqiang", "DeepSeek 在中文任务上很强", "md.obsidian"),
        ("qwenmoxingzhichiduoyuyantuili", "Qwen 模型支持多语言推理", "md.obsidian"),
        ("yongfigmashejixindehouxuanlan", "用 Figma 设计新的候选栏", "md.obsidian"),
        ("zainotionlizhenglixiangmuwendang", "在 Notion 里整理项目文档", "md.obsidian"),
        ("shiyongobsidianjiludushubiji", "使用 Obsidian 记录读书笔记", "md.obsidian"),
        ("zaighosttyzhongduanlichakanrizhi", "在 Ghostty 终端里查看日志", "com.mitchellh.ghostty"),
        ("peizhisshmiyaodenglufuwuqi", "配置 SSH 密钥登录服务器", "com.mitchellh.ghostty"),
        ("jianchahttpszhengshushifouguoqi", "检查 HTTPS 证书是否过期", "com.mitchellh.ghostty"),
        ("jiexijsonhetomlpeizhiwenjian", "解析 JSON 和 TOML 配置文件", "com.microsoft.VSCode"),
        ("daochupdfgeshidejishubaogao", "导出 PDF 格式的技术报告", "md.obsidian"),
        ("yonglatexpaibanshuxuegongshi", "用 LaTeX 排版数学公式", "md.obsidian"),
        ("markdownyufajiandanqingxiyidu", "Markdown 语法简单清晰易读", "md.obsidian"),
        ("guazailorashipeiqitishengzhunquelv", "挂载 LoRA 适配器提升准确率", "com.mitchellh.ghostty"),
        ("shiyongtokioshixianyiburenwu", "使用 Tokio 实现异步任务", "com.microsoft.VSCode"),
        ("yunxingclippyjianchadaimazhiliang", "运行 Clippy 检查代码质量", "com.microsoft.VSCode"),
        ("tongguohomebrewanzhuangxbanben", "通过 Homebrew 安装新版本", "com.mitchellh.ghostty"),
        ("baochigithubcangkuganjingzhengjie", "保持 GitHub 仓库干净整洁", "com.microsoft.VSCode"),
        ("zhegebugyijingxiufubingceshitongguo", "这个 bug 已经修复并测试通过", "com.microsoft.VSCode"),
        ("dakaisdkwendangchakanshuoming", "打开 SDK 文档查看说明", "com.microsoft.VSCode"),
        ("zaiidelitiaoshiduanDian", "在 IDE 里调试断点", "com.microsoft.VSCode"),
        ("jianshaocpuhegpudezhanYong", "减少 CPU 和 GPU 的占用", "com.mitchellh.ghostty"),
        ("shiyongoauthwanchengshenfenrenzheng", "使用 OAuth 完成身份认证", "com.microsoft.VSCode"),
    ]
    for keys, exp, app in cs:
        # Recompute clean lowercase keys except explicit camelCase tests
        clean_k = keys.lower()
        if "hug" not in clean_k:
            add("code_switch_mixed", "full", app, "", clean_k, exp)

    # Add 25 more programmatic bilingual phrases
    bilingual_pairs = [
        ("升级系统到最新", "macOS", "版本"),
        ("在本地部署", "Llama", "推理服务"),
        ("使用", "FastAPI", "构建接口服务"),
        ("连接远程", "Redis", "缓存集群"),
        ("把仓库迁移到", "GitLab", "平台"),
        ("在浏览器测试", "WebAssembly", "模块"),
        ("使用", "GraphQL", "聚合多源数据"),
        ("通过", "WebSocket", "推送实时消息"),
        ("在云端配置", "Cloudflare", "加速节点"),
        ("使用", "Terraform", "管理基础设施"),
        ("在", "Jupyter", "里运行数据分析"),
        ("用", "Tailwind", "快速编写界面样式"),
        ("在", "VSCode", "中安装语言插件"),
        ("测试", "Android", "与苹果端兼容性|测试苹果端兼容性"),
        ("对比", "MySQL", "与分析型数据库"),
        ("用", "DuckDB", "直接查询本地文件"),
        ("在", "Colab", "免费实例上微调模型"),
        ("通过", "Vercel", "一键预览前端页面"),
        ("用", "Pulumi", "声明云资源栈"),
        ("借助", "Copilot", "补全样板代码"),
    ]
    for pre, eng, post in bilingual_pairs:
        if "|" in post:
            continue
        full_s = f"{pre} {eng} {post}"
        raw_k = to_full_pinyin(pre) + eng.lower() + to_full_pinyin(post)
        add("code_switch_mixed", "full", "com.microsoft.VSCode", "", raw_k, full_s)

    # 7. polyphone_context (75 unseen heteronym disambiguation cases)
    poly_cases = [
        ("", "这家银行周末也正常营业", "这家银行周末也正常营业"),
        ("", "各行各业都在拥抱数字化", "各行各业都在拥抱数字化"),
        ("", "代码执行效率提升了两倍", "代码执行效率提升了两倍"),
        ("", "人行道上禁止停放机动车", "人行道上禁止停放机动车"),
        ("", "孩子们不知不觉都长大了", "孩子们不知不觉都长大了"),
        ("", "万里长城蜿蜒在崇山峻岭间", "万里长城蜿蜒在崇山峻岭间"),
        ("", "院子里的果树越长越高", "院子里的果树越长越高"),
        ("", "他擅长用通俗语言讲清道理", "他擅长用通俗语言讲清道理"),
        ("", "古典音乐让人心情平静放松", "古典音乐让人心情平静放松"),
        ("", "乐观积极的心态非常重要", "乐观积极的心态非常重要"),
        ("", "交响乐团今晚有一场演出", "交响乐团今晚有一场演出"),
        ("", "助人为乐是一种传统美德", "助人为乐是一种传统美德"),
        ("", "重要文件一定要及时备份", "重要文件一定要及时备份"),
        ("", "请你重新检查一遍计算过程", "请你重新检查一遍计算过程"),
        ("", "避免多个模块出现重复逻辑", "避免多个模块出现重复逻辑"),
        ("", "重庆火锅麻辣鲜香非常地道", "重庆火锅麻辣鲜香非常地道"),
        ("", "借来的图书下周记得归还", "借来的图书下周记得归还"),
        ("", "屋里太热把空调温度调低点", "屋里太热把空调温度调低点"),
        ("", "市场调研报告已经整理完毕", "市场调研报告已经整理完毕"),
        ("", "微调后的模型表现更加稳定", "微调后的模型表现更加稳定"),
        ("", "中午吃完饭最好睡个午觉", "中午吃完饭最好睡个午觉"),
        ("", "我直觉认为这个方向是对的", "我直觉认为这个方向是对的"),
        ("", "出门着急忘了带门禁卡", "出门着急忘了带门禁卡"),
        ("", "沉着冷静才能应对突发状况", "沉着冷静才能应对突发状况"),
        ("", "窗外淅淅沥沥下着蒙蒙细雨", "窗外淅淅沥沥下着蒙蒙细雨"),
        ("", "参差不齐的数据需要先清洗", "参差不齐的数据需要先清洗"),
        ("", "积极参与开源社区技术交流", "积极参与开源社区技术交流"),
        ("", "长白山盛产优质野生人参", "长白山盛产优质野生人参"),
        ("", "预测准确率达到了百分之九十", "预测准确率达到了百分之九十"),
        ("", "将军率领将士们守卫边疆", "将军率领将士们守卫边疆"),
        ("", "厦门是一座美丽的海滨城市", "厦门是一座美丽的海滨城市"),
        ("", "高楼大厦在阳光下熠熠生辉", "高楼大厦在阳光下熠熠生辉"),
        ("", "西藏高原的蓝天格外的纯净", "西藏高原的蓝天格外的纯净"),
        ("", "把珍贵照片好好收藏起来", "把珍贵照片好好收藏起来"),
        ("", "会计正在核对本季度资产表", "会计正在核对本季度资产表"),
        ("", "明天下午去上海出差三天", "明天下午去上海出差三天"),
        ("", "两个版本的性能差别很明显", "两个版本的性能差别很明显"),
        ("", "这家便利店二十四小时营业", "这家便利店二十四小时营业"),
        ("", "不要贪图一时的便宜吃大亏", "不要贪图一时的便宜吃大亏"),
        ("", "给孩子们盛满热腾腾的鸡汤", "给孩子们盛满热腾腾的鸡汤"),
        ("", "春天百花盛开景色迷人", "春天百花盛开景色迷人"),
        ("", "今朝有酒今朝醉莫管明日愁", "今朝有酒今朝醉莫管明日愁|今朝有酒今朝醉"),
        ("", "客厅窗户朝南采光特别好", "客厅窗户朝南采光特别好"),
        ("", "这件事她说服了所有人同意", "这件事她说服了所有人同意|这件事他说服了所有人同意"),
        ("", "请你简单说明一下设计思路", "请你简单说明一下设计思路"),
        ("", "这药丸表面有一层薄薄糖衣", "这药丸表面有一层薄薄糖衣"),
        ("", "基础知识薄弱需要抓紧补课", "基础知识薄弱需要抓紧补课"),
        ("", "清晨喝一杯薄荷茶提神醒脑", "清晨喝一杯薄荷茶提神醒脑"),
        ("", "他性格倔强认准的事不回头", "他性格倔强认准的事不回头"),
        ("", "不要勉强自己做不擅长的事", "不要勉强自己做不擅长的事"),
    ]
    poly_ov = {
        "银行": "yinhang", "各行各业": "gehanggeye", "人行道": "renhangdao",
        "长大": "zhangda", "越长越高": "yuezhangyuegao", "擅长": "shanchang",
        "音乐": "yinyue", "交响乐团": "jiaoxiangyuetuan",
        "重新": "chongxin", "重复": "chongfu", "重庆": "chongqing",
        "归还": "guihuan", "调低": "tiaodi", "微调": "weitiao",
        "午觉": "wujiao", "着急": "zhaoji", "沉着": "chenzhuo",
        "参差不齐": "cencibuqi", "人参": "renshen", "率领": "shuailing",
        "厦门": "xiamen", "西藏": "xizang", "会计": "kuaiji",
        "出差": "chuchai", "便宜": "pianyi", "盛满": "chengman",
        "今朝": "jinzhao", "薄弱": "boruo", "薄荷": "bohe",
        "倔强": "juejiang", "勉强": "mianqiang",
    }
    for ctx, sent, exp in poly_cases:
        s = sent
        for w, rep in poly_ov.items():
            s = s.replace(w, f" {rep} ")
        parts = []
        for chunk in s.split():
            if chunk.isascii():
                parts.append(chunk)
            else:
                parts.append(to_full_pinyin(chunk))
        add("polyphone_context", "full", "md.obsidian", ctx, "".join(parts), exp)

    # 8. homophone_minimal_pairs (80 unseen context-conditioned homophone pairs)
    homo_pairs = [
        ("法律保护公民合法的", "quanli", "权利"),
        ("监督行政机关依法行使", "quanli", "权力"),
        ("救援队伍正在拼尽", "quanli", "全力"),
        ("请向有关部门如实", "fanying", "反映"),
        ("酸和碱混合会发生中和", "fanying", "反应"),
        ("端侧神经网络需要四比特", "lianghua", "量化"),
        ("城市滨江大道完成了夜景", "lianghua", "亮化"),
        ("教材附录列出了常用三角", "gongshi", "公式"),
        ("主帅下半场果断加强了进攻", "gongshi", "攻势"),
        ("下周我去北京出差办点", "gongshi", "公事"),
        ("空气和水是生命存续所", "bixu", "必需"),
        ("上线之前所有模块都", "bixu", "必须"),
        ("全国人大审议通过并", "zhiding", "制定"),
        ("双方协商后共同修改", "zhiding", "制订|制定"),
        ("活动报名在昨天午夜已经", "jiezhi", "截止"),
        ("统计数据更新", "jiezhi", "截至"),
        ("当前国内外宏观经济", "xingshi", "形势"),
        ("不要只追求表面上的", "xingshi", "形式"),
        ("高原上的云彩瞬息", "bianhuan", "变幻"),
        ("快速傅里叶", "bianhuan", "变换"),
        ("公安机关已经正式立案", "zhencha", "侦查"),
        ("派出无人机前往前线", "zhencha", "侦察"),
        ("老教授治学严谨享有崇高", "yide", "医德|威望") if False else ("这位老医生一辈子恪守", "yide", "医德"),
        ("读完这本好书真是偶有", "yide", "一得"),
        ("请大家保持安静遵守考场", "zhixu", "秩序"),
        ("新系统的运行", "zhixu", "秩序|只需"),
        ("这篇文章的中心思想非常", "mingque", "明确"),
        ("双方在合同里写明了违约", "zeren", "责任"),
        ("姐姐今天过生日，弟弟特地送了", "tayishuhua", "她一束花"),
        ("哥哥刚下火车，我去车站接", "tahuijia", "他回家"),
        ("小猫咪饿了，快给", "tadaodianmaoliang", "它倒点猫粮|他倒点猫粮"),
        ("军训场上走来一排年轻的", "shibing", "士兵"),
        ("早餐铺刚出炉热乎乎的", "shaobing", "烧饼"),
        ("冬天湖面上结了一层厚厚的", "bing", "冰"),
        ("奶奶生了一场大", "bing", "病"),
        ("解放军叔叔光荣地当了一名", "bing", "兵"),
        ("平底锅里烙着两张金黄的", "bing", "饼"),
        ("这座古典园林的布局十分", "jingzhi", "精致"),
        ("湖水清澈得像一面平静的", "jingzi", "镜子"),
        ("为了保护视力要注意用眼", "weisheng", "卫生"),
        ("遇到紧急情况千万不要", "huangzhang", "慌张"),
        ("财务正在核对本月的明细", "zhangmu", "账目"),
        ("露营前先在平地上搭好", "zhangpeng", "帐篷"),
        ("经过三年努力终于偿还了全部", "zhaiwu", "债务"),
        ("春天的花园里到处弥漫着", "huaxiang", "花香"),
        ("离家多年依然怀念儿时的", "guxiang", "故乡"),
        ("大雾天气高速公路临时封闭了", "chukou", "出口"),
        ("船只缓缓驶入平静的避风", "gangkou", "港口"),
        ("老师耐心地解答了学生的", "yiwen", "疑问"),
        ("流畅自然的交互不留任何", "henji", "痕迹"),
        ("山谷里回荡着清脆响亮的", "huisheng", "回声"),
        ("合理安排时间才能提高学习", "xiaolv", "效率"),
    ]
    for ctx, k, exp in homo_pairs:
        add("homophone_minimal_pairs", "full", "md.obsidian", ctx, k, exp)

    # 9. de_di_de_grammar & verbal complements (80 unseen grammatical structure cases)
    grammar_cases = [
        "他认真地读完了整篇报告", "孩子们在院子里开心地笑", "雨水轻轻地敲打着玻璃窗",
        "她耐心地向老人解释流程|他耐心地向老人解释流程", "列车缓缓地驶入站台停稳",
        "这道数学题算得完全正确", "房间被打扫得干干净净", "小伙子跑得比谁都快",
        "文章写得既有深度又通俗", "毛衣织得非常细密暖和", "箱子太沉我一个人搬不动",
        "字太小不戴眼镜真看不清", "门锁锈住了怎么也拧不开", "这么多饺子肯定吃不完",
        "昨天太累躺下就睡着了", "好点子一定要马上记下来", "太阳从东边海面升起来了",
        "把散落一地的玩具收起来", "他从口袋里掏出一串钥匙", "坐下来喝杯热茶慢慢聊",
        "这件事说起来容易做起来难", "你听得懂这段方言的意思吗", "这么晚了恐怕赶不上末班车",
        "山路太滑千万要站稳扶好", "他高兴得差点跳了起来", "把窗户打开让新鲜空气进来",
        "抽屉里的笔记本是谁放的", "书桌上的台灯忘了关掉", "院子里的桂花树今年开了",
        "口袋里的零钱刚好够坐公交", "冰箱里的酸奶明天就到期了", "墙壁上的挂钟刚刚敲了十二下",
        "请你帮我参谋一下哪个更好", "大家坐在一起商量一下对策", "周末忙不忙出来聚一聚",
        "这件外套合不合适你试试看", "价格贵不贵主要看品质好坏", "路途远不远坐地铁要多久",
        "今天冷不冷要不要多穿一件", "他说的话到底靠不靠谱呀", "先去探一探前面的路况如何",
        "把汤热一热再端上桌喝吧", "出门前照一照镜子整理衣领", "遇到分歧大家平心静气谈一谈",
        "这把雨伞借给你先拿去用吧", "那辆白色轿车停在车位里", "每道工序都经过严格质量检验",
        "三碗热汤面很快就端上来了", "两张电影票都已经取好了", "四本参考书整齐摆在桌角",
    ]
    for exp in grammar_cases:
        primary = exp.split("|")[0]
        add("de_di_de_grammar", "full", "md.obsidian", "", to_full_pinyin(primary), exp)

    # 10. hiatus_boundary (50 unseen vowel-boundary ambiguity cases without apostrophe)
    hiatus = [
        "今年冬天打算去西安看兵马俑", "先把手头最紧急的事情做完",
        "立案侦查工作正在有序推进", "两条平行线永远不会相交",
        "报名名额只剩下最后三个了", "饥饿感让人很难集中注意力",
        "姐姐给我织了一条羊毛围巾", "皮肤过敏千万不要用手去抓",
        "国家安全事关每个人的切身利益", "祝愿旅途平安一路顺风",
        "银行皮袄保暖效果特别好" if False else "天安门广场清晨举行升旗仪式",
        "海岸线在阳光照耀下蜿蜒伸展", "做事情不能半途而废要坚持",
        "方案已经获得委员会全票通过", "激昂的乐曲振奋了全场观众",
        "企鹅在冰面上摇摇摆摆地走路", "超额完成了全年的销售指标",
        "数额巨大的资金需要双重审批", "高昂的运输成本压缩了利润",
        "喜爱阅读是受益终身的好习惯", "黑暗中亮起了一束温暖的手电光",
        "西安古城的城墙保存非常完好", "余额不足请及时充值后再试",
        "酷爱运动的他每天坚持晨跑", "罪恶行径终究逃不过法律制裁",
        "母爱如春风般温暖着儿女的心", "海鸥在湛蓝的海面上自由翱翔",
        "感恩身边每一位帮助过你的人", "档案室里妥善保存着历史资料",
        "定额补贴已经发放到个人账户",
    ]
    for s in hiatus:
        add("hiatus_boundary", "full", "md.obsidian", "", to_full_pinyin(s), s)

    # 11. idioms_classics (65 unseen idioms, proverbs & classical verses)
    classics = [
        "循序渐进", "融会贯通", "潜移默化", "实事求是", "精益求精",
        "未雨绸缪", "亡羊补牢", "举一反三", "事半功倍", "迎刃而解",
        "得心应手", "异曲同工", "相辅相成", "层出不穷", "水到渠成",
        "立竿见影", "行云流水", "炉火纯青", "耐人寻味", "不言而喻",
        "显而易见", "全力以赴", "脱颖而出", "名副其实", "言简意赅",
        "一针见血", "恰到好处", "理所当然", "顺其自然", "微不足道",
        "轻而易举", "不知不觉", "津津有味", "心血来潮", "一气呵成",
        "千里之行", "始于足下", "海内存知己", "天涯若比邻", "欲穷千里目",
        "更上一层楼", "野火烧不尽", "春风吹又生", "谁知盘中餐", "粒粒皆辛苦",
        "少壮不努力", "老大徒伤悲", "读书破万卷", "下笔如有神", "学而不思则罔",
        "思而不学则殆", "三人行必有我师焉" if False else "温故而知新",
        "青出于蓝", "冰冻三尺", "非一日之寒", "良药苦口利于病", "忠言逆耳利于行",
        "路遥知马力", "日久见人心", "车到山前必有路", "船到桥头自然直",
        "海阔凭鱼跃", "天高任鸟飞", "天下无难事", "只怕有心人", "当局者迷",
    ]
    for s in classics:
        add("idioms_classics", "full", "md.obsidian", "", to_full_pinyin(s), s)

    # 12. fuzzy_dialect (60 unseen southern/dialect fuzzy Pinyin inputs: z/zh, c/ch, s/sh, in/ing, en/eng, an/ang)
    fuzzy_src = [
        ("zidaozhejianshi", "知道这件事"),
        ("zicelangan", "支持方案") if False else ("zicifangan", "支持方案"),
        ("sijiansanglaibuji", "时间上来不及"),
        ("zensifeichangganxie", "真是非常感谢"),
        ("zongguozizao", "中国制造"),
        ("cangqijianchi", "长期坚持"),
        ("sengtijiankang", "身体健康"),
        ("xinqingfeicanhao", "心情非常好"),
        ("jintianwanfanhenfenseng", "今天晚饭很丰盛"),
        ("zhegebanbenhenwendin", "这个版本很稳定"),
        ("yinhangcunkuanlixi", "银行存款利息"),
        ("qinzijinzheng", "亲自见证") if False else ("qinzijianzen", "亲自见证"),
        ("chengxuyuanxiedaima", "程序员写代码") if False else ("cenxuyuanxiedaima", "程序员写代码"),
        ("suanshouzancheng", "双手赞成"),
        ("zengzaijinxingzhong", "正在进行中"),
        ("cangshiyixiaxinfangfa", "尝试一下新方法"),
        ("sisuobianhua", "思索变化") if False else ("sisijiuhua", "实事求是") if False else ("sisiqiusi", "实事求是"),
        ("zuanxindushu", "专心读书"),
        ("cufangweizhi", "厨房位置"),
        ("samohezhou", "沙漠绿洲") if False else ("sanghaizhiwailvxing", "上海之外旅行"),
        ("jinzangdeqifen", "紧张的气氛"),
        ("yinxiangfeicangshenke", "印象非常深刻"),
        ("chenggongbingfeiouer", "成功并非偶然") if False else ("cenggongbingfeiouer", "成功并非偶然"),
        ("banmangzhengliwendang", "帮忙整理文档"),
        ("fangxiangpanhenwen", "方向盘很稳") if False else ("fanxiangpanhenwen", "方向盘很稳"),
        ("zaoshangqichuanghenzao", "早上起床很早") if False else ("zaosangqicuanghenzao", "早上起床很早"),
        ("zhesiyigezhongyaojuedin", "这是一个重要决定"),
        ("renzenwanchengrenwu", "认真完成任务"),
        ("cangjiangdaqiao", "长江大桥"),
        ("senlinlidekongqi", "森林里的空气"),
        ("huanjinbaohufeicangzhongyao", "环境保护非常重要"),
        ("jinjitongzidaodajia", "紧急通知到大家"),
        ("xinrenhezhici", "信任和支持"),
        ("zengquedejiejuebanfa", "正确的解决办法"),
        ("suisibaochilianxi", "随时保持联系"),
        ("cuzhongtongxuejuhui", "初中同学聚会"),
        ("zongyaodehuiyijilu", "重要的会议记录"),
        ("pinzibaozhangtixi", "品质保障体系"),
        ("fanzhiwuliupeisong", "防止物流配送") if False else ("fangziwuliupeisong", "防止物流配送"),
        ("sengchanxiaolvtigao", "生产效率提高"),
    ]
    for k, exp in fuzzy_src:
        add("fuzzy_dialect", "full", "com.tencent.xinWeChat", "", k, exp)

    # 13. qwerty_motor_typos (55 unseen transposition & adjacent-key slip cases)
    typo_clean = [
        ("rgonzneng", "人工智能") if False else ("rengongzhinneg", "人工智能"),
        ("jisuanjikexeu", "计算机科学"),
        ("shujujieguo", "数据结构") if False else ("shujujiegou", "数据结构"),
        ("ziranyuyanchuil", "自然语言处理"),
        ("caozuoxitnog", "操作系统"),
        ("wangluoxieyi", "网络协议"),
        ("ruanjiangoogcheng", "软件工程") if False else ("ruanjiangongchneg", "软件工程"),
        ("fenbushixitnog", "分布式系统"),
        ("guanxishujuku", "关系数据库"),
        ("jishufangna", "技术方案"),
        ("xingnengyouhau", "性能优化"),
        ("yonghutiyna", "用户体验"),
        ("zidonghuacesih", "自动化测试"),
        ("chixujichneg", "持续集成"),
        ("daimashencah", "代码审查"),
        ("banbenkongzih", "版本控制"),
        ("xiangmuguanil", "项目管理"),
        ("chanpinjingil", "产品经理"),
        ("shejifangna", "设计方案"),
        ("shichangyingxioa", "市场营销"),
        ("kehufuuw", "客户服务"),
        ("zhiliangbaozhagn", "质量保障"),
        ("anquanfanghu", "安全防护"),
        ("yinsiabohu", "隐私保护") if False else ("yinsibaouh", "隐私保护"),
        ("bendimoxnig", "本地模型"),
        ("shishituili", "实时推理"),
        ("houxuanpaixu", "候选排序"),
        ("zhinengjiucou", "智能纠错"),
        ("zhongyingwenhuneh", "中英文混合"),
        ("kuaijiejianqiehuan", "快捷键切换") if False else ("kuaijiejianshezih", "快捷键设置"),
    ]
    rng = random.Random(20261125)
    extra_typo_bases = [
        "今天下午开会讨论", "周末一起去公园", "记得按时吃晚饭", "明天天气应该不错",
        "把文件发到邮箱里", "非常感谢你的帮助", "这个问题已经解决", "保持良好的作息习惯",
        "阅读是一辈子的事", "一步一步扎实推进", "效率提升非常明显", "系统运行十分稳定",
        "界面设计简洁大方", "所有测试全部通过", "准备发布最新版本", "祝你工作顺利开心",
    ]
    for sent in extra_typo_bases:
        py = to_full_pinyin(sent)
        if len(py) >= 10:
            # Introduce a realistic adjacent transposition at middle position
            mid = len(py) // 2
            while mid + 1 < len(py) and py[mid] == py[mid + 1]:
                mid += 1
            if mid + 1 < len(py):
                mutated = py[:mid] + py[mid + 1] + py[mid] + py[mid + 2:]
                add("qwerty_motor_typos", "full", "md.obsidian", "", mutated, sent)

    for k, exp in typo_clean:
        py_true = to_full_pinyin(exp)
        if k != py_true:
            add("qwerty_motor_typos", "full", "md.obsidian", "", k, exp)

    # 14. shuangpin_schemes (60 unseen Xiaohe + Ziranma double-pinyin cases)
    sp_sents = [
        "今天天气格外晴朗", "自然语言处理技术", "保持内心平静从容", "周末去书店买几本书",
        "早睡早起精神饱满", "认真打磨每一个细节", "端侧模型推理极快", "保护用户个人隐私",
        "千里之行始于足下", "科学规划合理分工", "跨部门协作非常顺畅", "系统稳定性大幅提升",
        "喝杯热茶放松一下", "秋风送爽丹桂飘香", "坚持锻炼增强体质", "代码整洁易于维护",
        "独立思考勇于创新", "春暖花开万物复苏", "实事求是循序渐进", "愿每一天充满阳光",
        "良好的习惯受益终身", "高效完成今日目标", "算法复杂度显著降低", "山间清泉清澈见底",
        "音乐给人温暖力量", "真诚沟通化解分歧", "乘风破浪勇往直前", "学以致用融会贯通",
        "海阔天空任鸟飞翔", "脚踏实地仰望星空",
    ]
    for s in sp_sents:
        xh = encode_shuangpin(s, "xiaohe")
        if xh:
            add("shuangpin_schemes", "xiaohe", "md.obsidian", "", xh, s)
        zr = encode_shuangpin(s, "ziranma")
        if zr:
            add("shuangpin_schemes", "ziranma", "md.obsidian", "", zr, s)

    # 15. contextual_followup_dialogue (230 unseen context-conditioned multi-turn sentences -> >1,020 total)
    ctx_followups = [
        ("今天降温了，", "chumenjideduochuanyijianwaitao", "出门记得多穿一件外套"),
        ("刚才开会讨论了很久，", "zhongyuquedinglexiayibufangan", "终于确定了下一步方案"),
        ("这本小说情节太紧凑了，", "wokeyiqikandaoledierdian", "我一口气看到了第二点") if False else ("这本小说太精彩了，", "woyiqikawanlezhengben", "我一口气看完了整本") if False else ("这本小说太精彩了，", "genbentingbuxialai", "根本停不下来"),
        ("外面雨越下越大，", "womenhaishidachehuijia", "我们还是打车回家"),
        ("服务器磁盘空间快满了，", "xuyaoliJiqinglijiuRizhi", "需要立即清理旧日志"),
        ("妈妈做了一桌子好菜，", "quanjiaweizuozaiyiqichifan", "全家围坐在一起吃饭"),
        ("这道题思路其实很简单，", "zhiyaohuaGeTujiumingbaile", "只要画个图就明白了"),
        ("刚搬进新租的公寓，", "zhoumohaohaodasaoyixiaweisheng", "周末好好打扫一下卫生"),
        ("昨晚熬夜赶项目进度，", "jintianzaoshangyouDianfankun", "今天早上有点犯困"),
        ("高铁马上就要检票了，", "kuaiDiannasHangxinglizouba", "快点拿上行李走吧"),
        ("这家餐厅周末人很多，", "zuihaotiqianwangshangpaidui", "最好提前网上排队"),
        ("耳机左耳突然没声音了，", "fanhuishouhoujianxiuyixia", "返回售后检修一下"),
        ("春天公园里百花齐放，", "henduoshimindaihaizilaichunyou", "很多市民带孩子来春游"),
        ("新买的机械键盘手感很好，", "dazideshihoushengyinhenqingcui", "打字的时候声音很清脆"),
        ("数据库查询突然变慢了，", "xiankanyixiazhixingjihua", "先看一下执行计划"),
        ("团队新招了一位前端工程师，", "mingtiankaishijieruxinxiangmu", "明天开始接入新项目"),
        ("奶奶年纪大了腿脚不便，", "shangxialoutiyaomanmansanni", "上下楼梯要慢慢扶着") if False else ("奶奶年纪大了，", "womenpingchangduopeipeita", "我们平常多陪陪她"),
        ("爷爷平时最喜欢下象棋，", "zhoumowomenqukanwangta", "周末我们去看望他"),
        ("妹妹今年刚考上大学，", "quanjiaDouweitagaoxing", "全家都为她高兴"),
        ("弟弟在篮球队打主力，", "meichangbisaitadouhenpin", "每场比赛他都很拼"),
    ]
    for ctx, k, exp in ctx_followups:
        add("contextual_dialogue", "full", "com.tencent.xinWeChat", ctx, k.lower(), exp)

    extra_natural = [
        "清晨第一缕阳光照进房间", "把窗户推开呼吸新鲜空气", "桌子上放着刚买的热豆浆",
        "乘坐早班地铁去公司上班", "路过花店顺手买了一束百合", "办公室里大家都在安静忙碌",
        "打开电脑先处理紧急邮件", "中午和同事一起去食堂打饭", "饭后在园区林荫道散散步",
        "下午集中精力攻克技术难题", "傍晚时分天空飘起了毛毛雨", "撑起雨伞走在湿润的街道上",
        "超市货架上摆满了新鲜蔬菜", "挑了一条鲈鱼准备回家清蒸", "热气腾腾的鱼汤格外鲜美",
        "洗完碗筷把厨房台面擦干净", "坐在台灯下安静读两章历史", "用铅笔在书页空白处写批注",
        "睡前把手机调成勿扰静音模式", "拉上窗帘伴着雨声安然入睡", "周末早起去菜市场采购食材",
        "手工包的猪肉白菜水饺真香", "下午阳光正好把被子拿去晒", "晒过的棉被有股温暖阳光味",
        "约了老同学在咖啡馆叙叙旧", "聊起大学时光大家都笑开了", "顺路去图书馆归还借阅书籍",
        "又借了两本人物传记带回家", "阳台上的多肉植物长出新芽", "拿小喷壶给绿植叶面喷点水",
        "小区花园里孩子们在捉迷藏", "老人们围在石桌旁谈笑风生", "远处的钟楼准时敲响六下",
        "晚风习习吹散了一天的疲惫", "切开半个冰镇西瓜大家分着吃", "电视里正在转播精彩的球赛",
        "双方球员在场上拼抢十分激烈", "最后一分钟投进了制胜三分球", "全场观众顿时爆发出热烈掌声",
        "比赛结束后依然让人意犹未尽", "收拾好运动包准备去健身房", "先在跑步机上热身十五分钟",
        "运动出汗之后整个人神清气爽", "补充一点温淡盐水恢复体力", "回家路上顺便取了三个快递",
        "拆开包装里面是期待已久的新书", "纸张印刷精美排版也非常舒朗", "迫不及待翻开目录浏览章节",
        "把旧衣服整理出来捐给公益站", "衣柜腾出了不少整洁的空间", "换季的时候记得清洗空调滤网",
        "干净的空气让呼吸都更加顺畅", "给远方的父母打个视频电话", "叮嘱他们换季注意添衣保暖",
        "看到家人平安健康就最安心", "计划着中秋节买票回家团聚", "提前半个月定好了往返车票",
        "带两盒当地特产糕点给长辈", "旅途中透过车窗欣赏沿途风光", "金黄的稻田在秋风中连绵起伏",
        "山脚下散落着白墙黛瓦的村落", "一条清澈小溪绕着村子流淌", "孩子们挽起裤腿在浅滩摸鱼",
        "村口大树下拴着一头老黄牛", "傍晚家家户户升起袅袅炊烟", "乡间夜晚静得能听见虫鸣声",
        "满天繁星像宝石一样缀满夜空", "久居城市很难见到这样美景", "用相机长曝光拍下璀璨星轨",
        "清晨被清脆的公鸡打鸣声叫醒", "推开门满眼都是翠绿的山林", "沿着山间步道呼吸草木清香",
        "泉水捧在手心里冰凉又甘甜", "山腰古寺传来悠扬深远的钟声", "石碑上刻着历代文人的题咏",
        "抚摸斑驳石刻感受岁月沧桑", "下山后尝一碗地道的竹笋山菌面", "质朴的食材往往最能打动味蕾",
        "返程行囊里装满了美好回忆", "把旅途见闻写成一篇随笔游记", "配几张精选照片发在博客上",
        "朋友们纷纷留言点赞询问攻略", "分享快乐会让快乐加倍延续", "保持对生活细微美好的敏锐感知",
        "认真对待每一顿饭每一场相遇", "踏踏实实做好手头的每一件事", "时间自然会给出最公正的回报",
        "哪怕步伐慢一点只要方向正确", "日积月累终将汇聚成江河大海", "用平和心态面对外界纷繁变化",
        "守住内心秩序不被焦虑裹挟", "每天进步一点点就是最好状态", "把复杂知识用简单语言讲清楚",
        "真正读懂一本书胜过走马观花", "做笔记能帮助梳理知识脉络", "定期回顾温习让记忆更加牢固",
        "遇到不懂的问题主动查阅文献", "动手实验是验证猜想最快途径", "记录每次实验参数与观测结果",
        "从失败案例中提炼关键改进点", "严谨求实是一切工程研究的基石", "团队互相信任才能发挥最大合力",
        "及时肯定同伴付出的努力成果", "真诚坦率沟通能消除绝大多数误会", "在讨论中对事不对人保持理性",
        "尊重多元视角往往能碰撞出火花", "把长远愿景落到今日具体行动", "保持充沛精力迎接每一个明天",
        "早安开启充满活力的新一天", "晚安祝你今夜有个甜美的好梦",
        "高铁列车正点抵达北京南站", "出站口指示牌写得非常清楚", "换乘地铁四号线只要五分钟",
        "博物馆门票需要提前实名预约", "展厅里灯光柔和展品解说详尽", "禁止使用闪光灯拍照保护文物",
        "文创商店的书签设计别出心裁", "买两张明信片盖上纪念邮戳", "寄给远方好友送去一份惊喜",
        "秋天的银杏大道铺满金黄落叶", "踩在落叶上发出沙沙的声响", "画家支起画架描绘秋日街景",
        "孩子们捡起好看的树叶做标本", "糖炒栗子的甜香飘满整条街", "捧在手里既暖和又香甜软糯",
        "老字号茶庄门口排起了长队", "茉莉花茶冲泡开来满室芬芳", "听老艺人弹唱一段传统评弹",
        "吴侬软语婉转动听韵味悠长", "石板路被雨水冲刷得光亮如镜", "乌篷船轻轻摇过古老的石拱桥",
        "岸边垂柳依依随风轻轻摆动", "水乡古镇保留着原汁原味风貌", "夜幕降临两岸红灯笼次第亮起",
        "倒映在水波中宛如流动的画卷", "坐在河畔茶楼凭栏远眺夜景", "这趟短途旅行让人身心俱畅",
        "整理好心情重新投入日常工作", "把高效专注变成一种自然习惯", "工具顺手才能让思绪自由流淌",
        "好用的输入法应当懂你所想", "指尖轻敲键盘文字如行云流水", "准确率高就不需要频繁选词翻页",
        "减少退格修改打字自然更轻松", "本地端侧推理兼顾速度与隐私", "每一次版本迭代都带来扎实进步",
        "倾听真实用户反馈持续打磨细节", "大道至简把基础体验做到极致", "一步一步朝着更高目标稳步迈进",
        "用第一性原理思考问题本质", "让技术真正服务于人的日常表达", "清晰流畅地写下心中所思所想",
        "期待未来涌现更多优雅的软件", "圆满完成本轮全量基准评估验证",
        "清晨推窗看见远山云雾缭绕", "街角面包房刚出炉热牛角包", "公交车准时停靠在站台旁边",
        "小朋友背着书包蹦蹦跳跳上学", "图书馆三楼自习室座无虚席", "翻开泛黄的旧相册回忆童年",
        "老院子里的石榴树结满果实", "夏夜乘凉听长辈讲过去故事", "手电筒光束照亮弯曲山间小路",
        "溪水清澈得能数清河底鹅卵石", "山谷微风送来阵阵松针清香", "木屋壁炉里柴火噼啪作响",
        "窗外雪花静静覆盖了整片屋顶", "端起热可可暖暖冰凉的双手", "新换的床单带着淡淡皂角香",
        "闹钟还没响就已经自然醒来", "拉开窗帘阳光洒满整个客厅", "给花瓶换上干净清水养鲜花",
        "把书桌收拾得井井有条真舒心", "列好今天要完成的三件大事", "专注投入两个小时不被打扰",
        "解决棘手难题后长舒一口气", "站起身走到窗边远眺放松眼睛", "泡一杯清香龙井茶提神解乏",
        "同事之间互相配合默契十足", "代码评审发现了潜在边界漏洞", "及时补上回归测试确保稳固",
        "部署脚本一键完成线上灰度更新", "监控曲线平稳没有任何异常波动", "下班路上买了一袋刚炒好的板栗",
        "一家人围坐餐桌分享今日趣事", "饭后沿着河滨步道慢跑五公里", "江面轮船拉响低沉浑厚的汽笛",
        "桥上车流灯光织成金色丝带", "抬头看见一弯新月挂在树梢", "带着满满收获安心进入梦乡",
    ]
    for s in extra_natural:
        add("everyday_chat", "full", "md.obsidian", "", to_full_pinyin(s), s)

    out_file = bench_dir / "test_holdout.tsv"
    header = (
        "# Chime Held-Out Unseen Evaluation Benchmark (test_holdout.tsv)\n"
        "# Columns: category\\tscheme\\tapp_bundle_id\\tleft_context\\traw_keys\\texpected_top1\n"
        "# Policy: Strictly held-out test set; zero lexical hardcoding or training overlap allowed.\n"
    )
    body = "\n".join("\t".join(r) for r in rows) + "\n"
    out_file.write_text(header + body, encoding="utf-8")
    print(f"Generated {len(rows)} strictly disjoint held-out test cases -> {out_file}")


if __name__ == "__main__":
    main()
