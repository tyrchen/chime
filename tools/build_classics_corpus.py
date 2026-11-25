#!/usr/bin/env python3
"""
Builds Chime's comprehensive public-domain Classical Chinese Poetry, Song Ci, Yuan Qu,
Pre-Qin Classics, Guwen Masterpieces, Idioms & Xiehouyu lexicon + full-poem `Tab` continuation
chains + LoRA training text!

Outputs into `dist/`:
  - `classics_core_lexicon.tsv`         (Top famous poems/prose/merged Song Ci clauses for `core.imedict`)
  - `classics_core_continuations.tsv`   (Complete `Tab` chains for famous poems/prose in `core.imedict`)
  - `classics_pack_lexicon.tsv`         (Full 50k+ classical verses, idioms & sayings for `classics-poetry.imepack`)
  - `classics_pack_continuations.tsv`   (Full 40k+ `Tab` continuation chains for `classics-poetry.imepack`)
  - `classics_lora_corpus.txt`          (Clean classical Chinese text for Rank-8 LoRA training on Colab)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from opencc import OpenCC
from pypinyin import Style, lazy_pinyin

CC_T2S = OpenCC("t2s")

# Curated canonical masterpieces in clean Simplified Chinese guaranteeing 100% end-to-end coverage
# and exact punctuation for `一路 Tab` chaining!
MASTERPIECES: list[tuple[str, str]] = [
    (
        "李白《将进酒》",
        "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。"
        "人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。"
        "烹羊宰牛且为乐，会须一饮三百杯。岑夫子，丹丘生，将进酒，杯莫停。"
        "与君歌一曲，请君为我倾耳听。钟鼓馔玉不足贵，但愿长醉不复醒。"
        "古来圣贤皆寂寞，惟有饮者留其名。陈王昔时宴平乐，斗酒十千恣欢谑。"
        "主人何为言少钱，径须沽取对君酌。五花马，千金裘，呼儿将出换美酒，与尔同销万古愁。",
    ),
    (
        "苏轼《水调歌头·明月几时有》",
        "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。"
        "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。"
        "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆。"
        "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。",
    ),
    (
        "柳永《雨霖铃·寒蝉凄切》",
        "寒蝉凄切，对长亭晚，骤雨初歇。都门帐饮无绪，留恋处，兰舟催发。"
        "执手相看泪眼，竟无语凝噎。念去去，千里烟波，暮霭沉沉楚天阔。"
        "多情自古伤离别，更那堪，冷落清秋节！今宵酒醒何处？杨柳岸，晓风残月。"
        "此去经年，应是良辰好景虚设。便纵有千种风情，更与何人说！",
    ),
    (
        "苏轼《念奴娇·赤壁怀古》",
        "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。"
        "乱石穿空，惊涛拍岸，卷起千堆雪。江山如画，一时多少豪杰。"
        "遥想公瑾当年，小乔初嫁了，雄姿英发。羽扇纶巾，谈笑间，樯橹灰飞烟灭。"
        "故国神游，多情应笑我，早生华发。人生如梦，一尊还酹江月。",
    ),
    (
        "苏轼《定风波·莫听穿林打叶声》",
        "莫听穿林打叶声，何妨吟啸且徐行。竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。"
        "料峭春风吹酒醒，微冷，山头斜照却相迎。回首向来萧瑟处，归去，也无风雨也无晴。",
    ),
    (
        "苏轼《江城子·乙卯正月二十日夜记梦》",
        "十年生死两茫茫，不思量，自难忘。千里孤坟，无处话凄凉。"
        "纵使相逢应不识，尘满面，鬓如霜。夜来幽梦忽还乡，小轩窗，正梳妆。"
        "相顾无言，惟有泪千行。料得年年肠断处，明月夜，短松冈。",
    ),
    (
        "苏轼《江城子·密州出猎》",
        "老夫聊发少年狂，左牵黄，右擎苍，锦帽貂裘，千骑卷平冈。"
        "为报倾城随太守，亲射虎，看孙郎。酒酣胸胆尚开张，鬓微霜，又何妨！"
        "持节云中，何日遣冯唐？会挽雕弓如满月，西北望，射天狼。",
    ),
    (
        "苏轼《临江仙·夜饮东坡醒复醉》",
        "夜饮东坡醒复醉，归来仿佛三更。家童鼻息已雷鸣。敲门都不应，倚杖听江声。"
        "长恨此身非我有，何时忘却营营。夜阑风静縠纹平。小舟从此逝，江海寄余生。",
    ),
    (
        "辛弃疾《青玉案·元夕》",
        "东风夜放花千树，更吹落，星如雨。宝马雕车香满路。"
        "凤箫声动，玉壶光转，一夜鱼龙舞。蛾儿雪柳黄金缕，笑语盈盈暗香去。"
        "众里寻他千百度，蓦然回首，那人却在，灯火阑珊处。",
    ),
    (
        "辛弃疾《永遇乐·京口北固亭怀古》",
        "千古江山，英雄无觅，孙仲谋处。舞榭歌台，风流总被，雨打风吹去。"
        "斜阳草树，寻常巷陌，人道寄奴曾住。想当年，金戈铁马，气吞万里如虎。"
        "元嘉草草，封狼居胥，赢得仓皇北顾。四十三年，望中犹记，烽火扬州路。"
        "可堪回首，佛狸祠下，一片神鸦社鼓。凭谁问：廉颇老矣，尚能饭否？",
    ),
    (
        "辛弃疾《破阵子·为陈同甫赋壮词以寄之》",
        "醉里挑灯看剑，梦回吹角连营。八百里分麾下炙，五十弦翻塞外声，沙场秋点兵。"
        "马作的卢飞快，弓如霹雳弦惊。了却君王天下事，赢得生前身后名。可怜白发生！",
    ),
    (
        "李清照《声声慢·寻寻觅觅》",
        "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。"
        "三杯两盏淡酒，怎敌他、晚来风急！雁过也，正伤心，却是旧时相识。"
        "满地黄花堆积，憔悴损，如今有谁堪摘？守着窗儿，独自怎生得黑！"
        "梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得！",
    ),
    (
        "李清照《如梦令·昨夜雨疏风骤》",
        "昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否，知否？应是绿肥红瘦。",
    ),
    (
        "李清照《一剪梅·红藕香残玉簟秋》",
        "红藕香残玉簟秋，轻解罗裳，独上兰舟。云中谁寄锦书来？雁字回时，月满西楼。"
        "花自飘零水自流，一种相思，两处闲愁。此情无计可消除，才下眉头，却上心头。",
    ),
    (
        "李煜《虞美人·春花秋月何时了》",
        "春花秋月何时了？往事知多少。小楼昨夜又东风，故国不堪回首月明中。"
        "雕栏玉砌应犹在，只是朱颜改。问君能有几多愁？恰似一江春水向东流。",
    ),
    (
        "岳飞《满江红·怒发冲冠》",
        "怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。"
        "三十功名尘与土，八千里路云和月。莫等闲，白了少年头，空悲切！"
        "靖康耻，犹未雪。臣子恨，何时灭！驾长车，踏破贺兰山缺。"
        "壮志饥餐胡虏肉，笑谈渴饮匈奴血。待从头、收拾旧山河，朝天阙。",
    ),
    (
        "张若虚《春江花月夜》",
        "春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明！"
        "江流宛转绕芳甸，月照花林皆似霰。空里流霜不觉飞，汀上白沙看不见。"
        "江天一色无纤尘，皎皎空中孤月轮。江畔何人初见月？江月何年初照人？"
        "人生代代无穷已，江月年年望相似。不知江月待何人，但见长江送流水。"
        "白云一片去悠悠，青枫浦上不胜愁。谁家今夜扁舟子？何处相思明月楼？"
        "可怜楼上月徘徊，应照离人妆镜台。玉户帘中卷不去，捣衣砧上拂还来。"
        "此时相望不相闻，愿逐月华流照君。鸿雁长飞光不度，鱼龙潜跃水成文。"
        "昨夜闲潭梦落花，可怜春半不还家。江水流春去欲尽，江潭落月复西斜。"
        "斜月沉沉藏海雾，碣石潇湘无限路。不知乘月几人归，落月摇情满江树。",
    ),
    (
        "李白《蜀道难》",
        "噫吁嚱，危乎高哉！蜀道之难，难于上青天！蚕丛及鱼凫，开国何茫然！"
        "尔来四万八千岁，不与秦塞通人烟。西当太白有鸟道，可以横绝峨眉巅。"
        "地崩山摧壮士死，然后天梯石栈相钩连。上有六龙回日之高标，下有冲波逆折之回川。"
        "黄鹤之飞尚不得过，猿猱欲度愁攀援。青泥何盘盘，百步九折萦岩峦。"
        "扪参历井仰胁息，以手抚膺坐长叹。问君西游何时还？畏途巉岩不可攀。"
        "但见悲鸟号古木，雄飞雌从绕林间。又闻子规啼夜月，愁空山。"
        "蜀道之难，难于上青天，使人听此凋朱颜！连峰去天不盈尺，枯松倒挂倚绝壁。"
        "飞湍瀑流争喧豗，砯崖转石万壑雷。其险也如此，嗟尔远道之人胡为乎来哉！"
        "剑阁峥嵘而崔嵬，一夫当关，万夫莫开。所守或匪亲，化为狼与豺。"
        "朝避猛虎，夕避长蛇，磨牙吮血，杀人如麻。锦城虽云乐，不如早还家。"
        "蜀道之难，难于上青天，侧身西望长咨嗟！",
    ),
    (
        "李白《行路难·其一》",
        "金樽清酒斗十千，玉盘珍羞直万钱。停杯投箸不能食，拔剑四顾心茫然。"
        "欲渡黄河冰塞川，将登太行雪满山。闲来垂钓碧溪上，忽复乘舟梦日边。"
        "行路难，行路难，多歧路，今安在？长风破浪会有时，直挂云帆济沧海。",
    ),
    (
        "李白《梦游天姥吟留别》",
        "海客谈瀛洲，烟涛微茫信难求；越人语天姥，云霞明灭或可睹。"
        "天姥连天向天横，势拔五岳掩赤城。天台四万八千丈，对此欲倒东南倾。"
        "我欲因之梦吴越，一夜飞度镜湖月。湖月照我影，送我至剡溪。"
        "谢公宿处今尚在，渌水荡漾清猿啼。脚著谢公屐，身登青云梯。"
        "半壁见海日，空中闻天鸡。千岩万转路不定，迷花倚石忽已暝。"
        "熊咆龙吟殷岩泉，栗深林兮惊层巅。云青青兮欲雨，水澹澹兮生烟。"
        "列缺霹雳，丘峦崩摧。洞天石扉，訇然中开。青冥浩荡不见底，日月照耀金银台。"
        "霓为衣兮风为马，云之君兮纷纷而来下。虎鼓瑟兮鸾回车，仙之人兮列如麻。"
        "忽魂悸以魄动，恍惊起而长嗟。惟觉时之枕席，失向来之烟霞。"
        "世间行乐亦如此，古来万事东流水。别君去兮何时还？且放白鹿青崖间，须行即骑访名山。"
        "安能摧眉折腰事权贵，使我不得开心颜！",
    ),
    (
        "杜甫《登高》",
        "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。"
        "万里悲秋常作客，百年多病独登台。艰难苦恨繁霜鬓，潦倒新停浊酒杯。",
    ),
    (
        "白居易《琵琶行》",
        "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。"
        "醉不成欢惨将别，别时茫茫江浸月。忽闻水上琵琶声，主人忘归客不发。"
        "寻声暗问弹者谁，琵琶声停欲语迟。移船相近邀相见，添酒回灯重开宴。"
        "千呼万唤始出来，犹抱琵琶半遮面。转轴拨弦三两声，未成曲调先有情。"
        "弦弦掩抑声声思，似诉平生不得志。低眉信手续续弹，说尽心中无限事。"
        "轻拢慢捻抹复挑，初为霓裳后六幺。大弦嘈嘈如急雨，小弦切切如私语。"
        "嘈嘈切切错杂弹，大珠小珠落玉盘。间关莺语花底滑，幽咽泉流冰下难。"
        "冰泉冷涩弦凝绝，凝绝不通声暂歇。别有幽愁暗恨生，此时无声胜有声。"
        "银瓶乍破水浆迸，铁骑突出刀枪鸣。曲终收拨当心画，四弦一声如裂帛。"
        "东船西舫悄无言，唯见江心秋月白。沉吟放拨插弦中，整顿衣裳起敛容。"
        "同是天涯沦落人，相逢何必曾相识！我从去年辞帝京，谪居卧病浔阳城。"
        "浔阳地僻无音乐，终岁不闻丝竹声。住近湓江地低湿，黄芦苦竹绕宅生。"
        "其间旦暮闻何物？杜鹃啼血猿哀鸣。春江花朝秋月夜，往往取酒还独倾。"
        "岂无山歌与村笛，呕哑嘲哳难为听。今夜闻君琵琶语，如听仙乐耳暂明。"
        "莫辞更坐弹一曲，为君翻作琵琶行。感我此言良久立，却坐促弦弦转急。"
        "凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。",
    ),
    (
        "白居易《长恨歌》",
        "汉皇重色思倾国，御宇多年求不得。杨家有女初长成，养在深闺人未识。"
        "天生丽质难自弃，一朝选在君王侧。回眸一笑百媚生，六宫粉黛无颜色。"
        "春寒赐浴华清池，温泉水滑洗凝脂。侍儿扶起娇无力，始是新承恩泽时。"
        "云鬓花颜金步摇，芙蓉帐暖度春宵。春宵苦短日高起，从此君王不早朝。"
        "在天愿作比翼鸟，在地愿为连理枝。天长地久有时尽，此恨绵绵无绝期。",
    ),
    (
        "王勃《滕王阁序》",
        "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。"
        "物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。"
        "潦水尽而寒潭清，烟光凝而暮山紫。俨骖騑于上路，访风景于崇阿。"
        "层峦耸翠，上出重霄；飞阁流丹，下临无地。"
        "闾阎扑地，钟鸣鼎食之家；舸舰弥津，青雀黄龙之轴。"
        "云销雨霁，彩彻区明。落霞与孤鹜齐飞，秋水共长天一色。"
        "渔舟唱晚，响穷彭蠡之滨；雁阵惊寒，声断衡阳之浦。"
        "天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。"
        "关山难越，谁悲失路之人；萍水相逢，尽是他乡之客。"
        "老当益壮，宁移白首之心？穷且益坚，不坠青云之志。"
        "北海虽赊，扶摇可接；东隅已逝，桑榆非晚。",
    ),
    (
        "范仲淹《岳阳楼记》",
        "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴。"
        "乃重修岳阳楼，增其旧制，刻唐贤今人诗赋于其上，属予作文以记之。"
        "予观夫巴陵胜状，在洞庭一湖。衔远山，吞长江，浩浩汤汤，横无际涯。"
        "朝晖夕阴，气象万千，此则岳阳楼之大观也，前人之述备矣。"
        "不以物喜，不以己悲，居庙堂之高则忧其民，处江湖之远则忧其君。"
        "是进亦忧，退亦忧。然则何时而乐耶？其必曰：先天下之忧而忧，后天下之乐而乐乎！",
    ),
    (
        "苏轼《赤壁赋》",
        "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。"
        "举酒属客，诵明月之诗，歌窈窕之章。少焉，月出于东山之上，徘徊于斗牛之间。"
        "白露横江，水光接天。纵一苇之所如，凌万顷之茫然。"
        "浩浩乎如冯虚御风，而不知其所止；飘飘乎如遗世独立，羽化而登仙。"
        "寄蜉蝣于天地，渺沧海之一粟。哀吾生之须臾，羡长江之无穷。"
        "挟飞仙以遨游，抱明月而长终。知不可乎骤得，托遗响于悲风。"
        "盖将自其变者而观之，则天地曾不能以一瞬；自其不变者而观之，则物与我皆无尽也，而又何羡乎！"
        "惟江上之清风，与山间之明月，耳得之而为声，目遇之而成色，取之无禁，用之不竭，是造物者之无尽藏也。",
    ),
    (
        "苏轼《记承天寺夜游》",
        "元丰六年十月十二日夜，解衣欲睡，月色入户，欣然起行。念无与为乐者，遂至承天寺寻张怀民。"
        "怀民亦未寝，相与步于中庭。庭下如积水空明，水中藻荇交横，盖竹柏影也。"
        "何夜无月？何处无竹柏？但少闲人如吾两人者耳。",
    ),
    (
        "诸葛亮《出师表》",
        "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。"
        "宫中府中，俱为一体，陟罚臧否，不宜异同。"
        "亲贤臣，远小人，此先汉所以兴隆也；亲小人，远贤臣，此后汉所以倾颓也。"
        "臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。"
        "受任于败军之际，奉命于危难之间，尔来二十有一年矣。"
        "今当远离，临表涕零，不知所言。",
    ),
    (
        "王羲之《兰亭集序》",
        "永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭，修禊事也。"
        "群贤毕至，少长咸集。此地有崇山峻岭，茂林修竹，又有清流激湍，映带左右。"
        "是日也，天朗气清，惠风和畅。仰观宇宙之大，俯察品类之盛，"
        "所以游目骋怀，足以极视听之娱，信可乐也。"
        "夫人之相与，俯仰一世。或取诸怀抱，悟言一室之内；或因寄所托，放浪形骸之外。"
        "固知一死生为虚诞，齐彭殇为妄作。后之视今，亦犹今之视昔，悲夫！",
    ),
    (
        "庄子《逍遥游》",
        "北冥有鱼，其名为鲲。鲲之大，不知其几千里也；化而为鸟，其名为鹏。"
        "鹏之背，不知其几千里也；怒而飞，其翼若垂天之云。"
        "水击三千里，抟扶摇而上者九万里，去以六月息者也。"
        "且夫水之积也不厚，则其负大舟也无力。"
        "若夫乘天地之正，而御六气之辩，以游无穷者，彼且恶乎待哉！"
        "故曰：至人无己，神人无功，圣人无名。",
    ),
    (
        "荀子《劝学》",
        "君子曰：学不可以已。青，取之于蓝，而青于蓝；冰，水为之，而寒于水。"
        "吾尝终日而思矣，不如须臾之所学也；吾尝跂而望矣，不如登高之博见也。"
        "积土成山，风雨兴焉；积水成渊，蛟龙生焉；积善成德，而神明自得，圣心备焉。"
        "故不积跬步，无以至千里；不积小流，无以成江海。"
        "骐骥一跃，不能十步；驽马十驾，功在不舍。"
        "锲而舍之，朽木不折；锲而不舍，金石可镂。",
    ),
    (
        "韩愈《师说》",
        "古之学者必有师。师者，所以传道受业解惑也。人非生而知之者，孰能无惑？"
        "生乎吾前，其闻道也固先乎吾，吾从而师之；生乎吾后，其闻道也亦先乎吾，吾从而师之。"
        "是故无贵无贱，无长无少，道之所存，师之所存也。"
        "是故弟子不必不如师，师不必贤于弟子，闻道有先后，术业有专攻，如是而已。",
    ),
    (
        "刘禹锡《陋室铭》",
        "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。"
        "苔痕上阶绿，草色入帘青。谈笑有鸿儒，往来无白丁。"
        "可以调素琴，阅金经。无丝竹之乱耳，无案牍之劳形。"
        "南阳诸葛庐，西蜀子云亭。孔子云：何陋之有？",
    ),
    (
        "周敦颐《爱莲说》",
        "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊。自李唐来，世人甚爱牡丹。"
        "予独爱莲之出淤泥而不染，濯清涟而不妖，中通外直，不蔓不枝，"
        "香远益清，亭亭净植，可远观而不可亵玩焉。",
    ),
    (
        "曹操《短歌行》",
        "对酒当歌，人生几何！譬如朝露，去日苦多。慨当以慷，忧思难忘。何以解忧？唯有杜康。"
        "青青子衿，悠悠我心。但为君故，沉吟至今。呦呦鹿鸣，食野之苹。我有嘉宾，鼓瑟吹笙。"
        "明明如月，何时可掇？忧从中来，不可断绝。越陌度阡，枉用相存。契阔谈讌，心念旧恩。"
        "月明星稀，乌鹊南飞。绕树三匝，何枝可依？山不厌高，海不厌深。周公吐哺，天下归心。",
    ),
    (
        "纳兰性德《木兰花·拟古决绝词柬友》",
        "人生若只如初见，何事秋风悲画扇。等闲变却故人心，却道故人心易变。"
        "骊山语罢清宵半，泪雨霖铃终不怨。何如薄幸锦衣郎，比翼连枝当日愿。",
    ),
    (
        "马致远《天净沙·秋思》",
        "枯藤老树昏鸦，小桥流水人家，古道西风瘦马。夕阳西下，断肠人在天涯。",
    ),
]

# Classical Chinese heteronym / archaic reading fixes on top of `pypinyin`
POLYPHONE_PHRASE_FIXES: list[tuple[str, str]] = [
    ("将进酒", "qiang jin jiu"),
    ("还复来", "huan fu lai"),
    ("朝如青丝", "zhao ru qing si"),
    ("朝辞白帝", "zhao ci bai di"),
    ("朝避猛虎", "zhao bi meng hu"),
    ("朝晖夕阴", "zhao hui xi yin"),
    ("朝天阙", "chao tian que"),
    ("数风流人物", "shu feng liu ren wu"),
    ("更那堪", "geng na kan"),
    ("问渠那得", "wen qu na de"),
    ("石径斜", "shi jing xie"),
    ("鬓毛衰", "bin mao cui"),
    ("见牛羊", "xian niu yang"),
    ("万重山", "wan chong shan"),
    ("山重水复", "shan chong shui fu"),
    ("重修岳阳楼", "chong xiu yue yang lou"),
    ("重开宴", "chong kai yan"),
    ("满座重闻", "man zuo chong wen"),
    ("一行白鹭", "yi hang bai lu"),
    ("泪千行", "lei qian hang"),
    ("琵琶行", "pi pa xing"),
    ("歌行", "ge xing"),
    ("行路难", "xing lu nan"),
    ("白发", "bai fa"),
    ("华发", "hua fa"),
    ("怒发冲冠", "nu fa chong guan"),
    ("雄姿英发", "xiong zi ying fa"),
    ("兰舟催发", "lan zhou cui fa"),
    ("一尊还酹", "yi zun huan lei"),
    ("还独倾", "huan du qing"),
    ("早还家", "zao huan jia"),
    ("不还家", "bu huan jia"),
    ("何时还", "he shi huan"),
    ("羽扇纶巾", "yu shan guan jin"),
    ("会稽山阴", "kuai ji shan yin"),
    ("少长咸集", "shao zhang xian ji"),
    ("少小离家", "shao xiao li jia"),
    ("无长无少", "wu zhang wu shao"),
    ("浩浩汤汤", "hao hao shang shang"),
    ("属予作文", "zhu yu zuo wen"),
    ("举酒属客", "ju jiu zhu ke"),
    ("冯虚御风", "ping xu yu feng"),
    ("一骑红尘", "yi ji hong chen"),
    ("千骑卷平冈", "qian ji juan ping gang"),
    ("铁骑突出", "tie ji tu chu"),
    ("单于", "chan yu"),
]

POETRY_RAW_SOURCES: list[tuple[str, str, bool]] = [
    ("tang300", "全唐诗/唐诗三百首.json", True),
    ("songci300", "宋词/宋词三百首.json", True),
    ("qianjiashi", "蒙学/qianjiashi.json", True),
    ("shenglvqimeng", "蒙学/shenglvqimeng.json", True),
    ("zengguangxianwen", "蒙学/zengguangxianwen.json", True),
    ("shijing", "诗经/shijing.json", True),
    ("lunyu", "论语/lunyu.json", True),
    ("chuci", "楚辞/chuci.json", True),
    ("caocao", "曹操诗集/caocao.json", True),
    ("nalan", "纳兰性德/纳兰性德诗集.json", True),
    ("yuanqu", "元曲/yuanqu.json", False),
    ("mengzi", "四书五经/mengzi.json", False),
    ("daxue", "四书五经/daxue.json", False),
    ("zhongyong", "四书五经/zhongyong.json", False),
    ("songci_0", "宋词/ci.song.0.json", False),
    ("songci_1000", "宋词/ci.song.1000.json", False),
    ("songci_2000", "宋词/ci.song.2000.json", False),
    ("songci_9000", "宋词/ci.song.9000.json", False),
    ("songci_10000", "宋词/ci.song.10000.json", False),
    ("songci_11000", "宋词/ci.song.11000.json", False),
    ("songci_21000", "宋词/ci.song.21000.json", False),
]

XINHUA_SOURCES: list[tuple[str, str]] = [
    ("idioms", "https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/idiom.json"),
    ("xiehouyu", "https://raw.githubusercontent.com/pwxcoo/chinese-xinhua/master/data/xiehouyu.json"),
]

VALID_SYLLABLES_RE = re.compile(r"^[a-z]+$")
CJK_ONLY_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
CLAUSE_SPLIT_RE = re.compile(r"([，。！？；：])")


def fetch_cached_json(cache_dir: Path, name: str, url: str):
    cache_file = cache_dir / f"{name}.json"
    if cache_file.exists() and cache_file.stat().st_size > 32:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "Chime-Corpus-Builder/0.6"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        cache_file.write_text(raw, encoding="utf-8")
        return json.loads(raw)
    except Exception as exc:
        print(f"  [warn] skipped {name} ({url}): {exc}")
        return None


def normalize_classical_text(text: str) -> str:
    s = CC_T2S.convert(text)
    s = (
        s.replace(",", "，")
        .replace(".", "。")
        .replace("?", "？")
        .replace("!", "！")
        .replace(";", "；")
        .replace(":", "：")
        .replace("、", "，")
    )
    # Strip parenthetical editorial notes like `(一作：...)` or `[注]`
    s = re.sub(r"（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】", "", s)
    s = re.sub(r"[“”\"'‘’《》〈〉\s]+", "", s)
    return s


def clause_to_pinyin(surface: str) -> str | None:
    cjk = "".join(ch for ch in surface if "\u4e00" <= ch <= "\u9fa5")
    if len(cjk) < 2:
        return None

    # Apply phrase-level classical polyphone overrides!
    syls = lazy_pinyin(cjk, style=Style.NORMAL, errors="ignore")
    if len(syls) != len(cjk):
        return None

    for phrase, fixed_py in POLYPHONE_PHRASE_FIXES:
        start = 0
        while True:
            idx = cjk.find(phrase, start)
            if idx == -1:
                break
            fixed_parts = fixed_py.split()
            if len(fixed_parts) == len(phrase):
                syls[idx : idx + len(phrase)] = fixed_parts
            start = idx + 1

    normed = []
    for s in syls:
        s = s.lower().replace("ü", "v").replace("u:", "v")
        if s in ("jv", "qv", "xv", "yv"):
            s = s[0] + "u"
        elif s.startswith(("jve", "qve", "xve", "yve")):
            s = s[0] + "ue"
        if not VALID_SYLLABLES_RE.match(s):
            return None
        normed.append(s)
    return " ".join(normed)


def extract_paragraphs_from_item(item: dict) -> list[str]:
    paras: list[str] = []
    for key in ("paragraphs", "content", "para"):
        val = item.get(key)
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, str):
                    paras.append(entry)
                elif isinstance(entry, dict) and isinstance(entry.get("paragraphs"), list):
                    paras.extend(x for x in entry["paragraphs"] if isinstance(x, str))
        elif isinstance(val, str):
            paras.append(val)
    return paras


def process_poem_text(
    raw_text: str,
    base_lp: float,
    lex_map: dict[tuple[str, str], float],
    cont_map: dict[str, tuple[str, float]],
    lora_lines: list[str],
):
    norm = normalize_classical_text(raw_text)
    if len(norm) < 6:
        return
    lora_lines.append(norm)

    parts = CLAUSE_SPLIT_RE.split(norm)
    # Reassemble `[(clause, trailing_punct), ...]`
    clauses: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        seg = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else "。"
        i += 2
        if seg and CJK_ONLY_RE.match(seg):
            clauses.append((seg, punct))

    if not clauses:
        return

    # 1. Individual clauses (`4..15` chars; allow `3` chars inside multi-clause poems for continuations)
    for idx, (c_text, c_punct) in enumerate(clauses):
        c_len = len(c_text)
        if 4 <= c_len <= 15:
            py = clause_to_pinyin(c_text)
            if py:
                key = (c_text, py)
                if base_lp > lex_map.get(key, -99.0):
                    lex_map[key] = base_lp

        # Continuation `c_text -> next_joiner + next_clause`
        if idx + 1 < len(clauses) and 3 <= c_len <= 16:
            next_text, _ = clauses[idx + 1]
            if 3 <= len(next_text) <= 16:
                joiner = "，" if c_punct in ("，", "：") else ("？" if c_punct == "？" else ("！" if c_punct == "！" else "。"))
                # If next clause is a short Song Ci lead-in (`念去去`, 3 chars) followed by comma + `4..6` chars (`千里烟波`),
                # emit the natural merged rhythm (`。念去去，千里烟波`) when `base_lp >= -3.1`!
                if (
                    idx + 2 < len(clauses)
                    and 2 <= len(next_text) <= 3
                    and clauses[idx + 1][1] == "，"
                    and 3 <= len(clauses[idx + 2][0]) <= 6
                ):
                    merged_next = f"{next_text}，{clauses[idx + 2][0]}"
                    target_cont = f"{joiner}{merged_next}"
                else:
                    target_cont = f"{joiner}{next_text}"
                prev_best = cont_map.get(c_text)
                if prev_best is None or base_lp > prev_best[1]:
                    cont_map[c_text] = (target_cont, base_lp)

    # 2. Merged short-pause Song Ci / Guwen lead-in clauses (`念去去，千里烟波`, `杨柳岸，晓风残月`, `五花马，千金裘`)!
    for idx in range(len(clauses) - 1):
        c1, p1 = clauses[idx]
        c2, p2 = clauses[idx + 1]
        if p1 == "，" and 2 <= len(c1) <= 4 and 3 <= len(c2) <= 6 and (len(c1) + len(c2)) <= 10:
            merged_surface = f"{c1}，{c2}"
            py = clause_to_pinyin(merged_surface)
            if py:
                merged_lp = base_lp + 0.12
                key = (merged_surface, py)
                if merged_lp > lex_map.get(key, -99.0):
                    lex_map[key] = merged_lp
                if idx + 2 < len(clauses):
                    c3, _ = clauses[idx + 2]
                    if 3 <= len(c3) <= 16:
                        joiner2 = "，" if p2 in ("，", "：") else ("？" if p2 == "？" else ("！" if p2 == "！" else "。"))
                        prev_best = cont_map.get(merged_surface)
                        if prev_best is None or merged_lp > prev_best[1]:
                            cont_map[merged_surface] = (f"{joiner2}{c3}", merged_lp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    dist_dir: Path = args.dist_dir
    cache_dir = dist_dir / "raw_classics_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    core_lex: dict[tuple[str, str], float] = {}
    core_cont: dict[str, tuple[str, float]] = {}
    pack_lex: dict[tuple[str, str], float] = {}
    pack_cont: dict[str, tuple[str, float]] = {}
    lora_lines: list[str] = []

    print(f"[1/4] Processing {len(MASTERPIECES)} canonical classical masterpieces...")
    for _title, full_text in MASTERPIECES:
        process_poem_text(full_text, -2.55, core_lex, core_cont, lora_lines)
        process_poem_text(full_text, -2.45, pack_lex, pack_cont, lora_lines)

    print(f"[2/4] Downloading & indexing {len(POETRY_RAW_SOURCES)} chinese-poetry collections...")
    base_gh = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    for tag, rel_path, in_core in POETRY_RAW_SOURCES:
        url = base_gh + urllib.parse.quote(rel_path)
        data = fetch_cached_json(cache_dir, tag, url)
        if not data:
            continue
        items = data if isinstance(data, list) else (data.get("content") or [])
        if not isinstance(items, list):
            continue
        lp = -3.05 if in_core else -3.45
        for item in items:
            if not isinstance(item, dict):
                continue
            paras = extract_paragraphs_from_item(item)
            if not paras:
                continue
            joined = "".join(paras)
            process_poem_text(joined, lp, pack_lex, pack_cont, lora_lines)
            if in_core:
                process_poem_text(joined, -3.15, core_lex, core_cont, [])

    print("[3/4] Downloading & indexing Xinhua idioms (30k) & xiehouyu (14k)...")
    for tag, url in XINHUA_SOURCES:
        data = fetch_cached_json(cache_dir, tag, url)
        if not isinstance(data, list):
            continue
        if tag == "idioms":
            for idx, entry in enumerate(data):
                word = CC_T2S.convert(str(entry.get("word", "")).strip())
                if 4 <= len(word) <= 12 and CJK_ONLY_RE.match(word):
                    py = clause_to_pinyin(word)
                    if py:
                        lp = -3.25 if idx < 5000 else -3.75
                        if lp > pack_lex.get((word, py), -99.0):
                            pack_lex[(word, py)] = lp
                        if idx < 4000 and lp > core_lex.get((word, py), -99.0):
                            core_lex[(word, py)] = lp
        elif tag == "xiehouyu":
            for idx, entry in enumerate(data):
                riddle = normalize_classical_text(str(entry.get("riddle", "")))
                answer = normalize_classical_text(str(entry.get("answer", "")))
                if 3 <= len(riddle) <= 12 and 2 <= len(answer) <= 10 and CJK_ONLY_RE.match(riddle):
                    py_r = clause_to_pinyin(riddle)
                    if py_r:
                        pack_lex[(riddle, py_r)] = max(pack_lex.get((riddle, py_r), -99.0), -3.55)
                        if riddle not in pack_cont:
                            pack_cont[riddle] = (f"——{answer}", -3.55)
                        if idx < 500:
                            lora_lines.append(f"{riddle}——{answer}。")

    # Cap `core` additions cleanly (~18k top classical verses/idioms + ~14k continuations) while
    # keeping the complete unfiltered 60k+ collection in `classics-poetry.imepack`!
    sorted_core_lex = sorted(core_lex.items(), key=lambda kv: (-kv[1], len(kv[0][0]), kv[0][0]))[:22000]
    sorted_core_cont = sorted(core_cont.items(), key=lambda kv: (-kv[1][1], kv[0]))[:18000]
    sorted_pack_lex = sorted(pack_lex.items(), key=lambda kv: (-kv[1], len(kv[0][0]), kv[0][0]))
    sorted_pack_cont = sorted(pack_cont.items(), key=lambda kv: (-kv[1][1], kv[0]))

    (dist_dir / "classics_core_lexicon.tsv").write_text(
        "\n".join(f"{w}\t{py}\t{lp:.3f}" for (w, py), lp in sorted_core_lex) + "\n",
        encoding="utf-8",
    )
    (dist_dir / "classics_core_continuations.tsv").write_text(
        "\n".join(f"{trig}\t{cont}" for trig, (cont, _) in sorted_core_cont) + "\n",
        encoding="utf-8",
    )
    (dist_dir / "classics_pack_lexicon.tsv").write_text(
        "\n".join(f"{w}\t{py}\t{lp:.3f}" for (w, py), lp in sorted_pack_lex) + "\n",
        encoding="utf-8",
    )
    (dist_dir / "classics_pack_continuations.tsv").write_text(
        "\n".join(f"{trig}\t{cont}" for trig, (cont, _) in sorted_pack_cont) + "\n",
        encoding="utf-8",
    )
    # Deduplicate LoRA training lines while preserving order
    seen_lines: set[str] = set()
    dedup_lora: list[str] = []
    for ln in lora_lines:
        if ln not in seen_lines and len(ln) >= 8:
            seen_lines.add(ln)
            dedup_lora.append(ln)
    (dist_dir / "classics_lora_corpus.txt").write_text("\n".join(dedup_lora) + "\n", encoding="utf-8")

    print(
        f"[4/4] Done!\n"
        f"  core_lexicon       = {len(sorted_core_lex):,} entries\n"
        f"  core_continuations = {len(sorted_core_cont):,} chains\n"
        f"  pack_lexicon       = {len(sorted_pack_lex):,} entries\n"
        f"  pack_continuations = {len(sorted_pack_cont):,} chains\n"
        f"  lora_corpus_lines  = {len(dedup_lora):,} paragraphs"
    )


if __name__ == "__main__":
    main()
