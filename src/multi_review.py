#!/usr/bin/env python3
"""跨家族多模型独立审查 + 红队。每个(题目,角色,模型)一次独立调用，结果落 JSON。"""
import json, pathlib, sys, time, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

ROOT = pathlib.Path("/home/opsuser/.hermes/profiles/instance2/work/cumcm-2026")
PROF = pathlib.Path("/home/opsuser/.hermes/profiles/instance2")
OUT = ROOT / "results" / "multi_review.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((PROF / "config.yaml").read_text())
gw = cfg["providers"]["kiro-gateway"]
client = OpenAI(base_url=gw["base_url"], api_key=gw["api_key"], timeout=600.0)

# 题面（已由 pdftotext 提取，此处存要点+硬数据，避免超长）
PROBLEMS = {
    "A": """CUMCM2026 A题 药材的烘干问题（热风烘干：预热平衡 + 恒温干燥两阶段）
圆柱形药材：长25cm，半径2cm。初始温度28°C，初始水分浓度(干基含水率)2.55 kg/kg。
附件1: 烘干初期烘房温度(°C)与水分浓度(kg/kg)时序，0~14400s，步长60s，共241行。
  实测: 温度28→50.2°C(约3600s后稳定在≈50°C波动)，烘房水分浓度0.0196→0.0499 kg/kg。
附件2: 烘干过程药材半径(cm)，0~259200s(72h)，步长1800s，共145行。
  实测: 半径 2.000cm 单调下降至 1.198cm（收缩40.1%），最后变化在 t=241200s(67.0h)，之后恒定。
问题1: 建预热平衡阶段温度与水分浓度变化规律模型。给出100/300/600/900/1200/1500/1800s，
  距中心0/0.5/1/1.5/2cm的结果；完整结果(1800s内每1s、每0.1cm)存 result1.xlsx。保留4位小数。
  附录2参数: 密度820 kg/m3, 比热容2600 J/(kg·K), 热导率0.36 W/(m·K),
    对流换热系数25 W/(m2·K), 对流传质系数8e-7 m/s, D=7e-9*exp(-0.89/C) m2/s。
问题2: 整个烘干过程(2-3天)温度与水分浓度模型。3h内每0.5h结果；完整每1s/0.1cm存result2.xlsx。
  附录3经验公式: rho=650+128C; cp=1450+2736*C/(C+1); k=0.21+0.38*C/(C+1);
    D=2.4e-3*exp(-0.45/C)*exp(-3850/T)  (T为K)
问题3: 各处水分浓度需低于0.15 kg/kg，确定烘干所需时间(h)。每6h/每0.5cm表；
  完整每60s/0.1cm存result3.xlsx。
问题4: 考虑水分流失导致尺寸变化(按附件2)，确定烘干时长。
  附录4经验公式: rho=760+90C; cp=1850+2150*C/(C+1); k=0.12+0.20*C/(C+1);
    D=4.2e-4*exp(-0.30/C)*exp(-3850/T)
  结果表列头为"到药材中心距离 0/0.5/.../药材表面"，完整每60s/0.1cm存result4.xlsx。""",

    "B": """CUMCM2026 B题 无线电干扰源的快速自动定位与清除
目标区域: 半径1800m圆域，圆心为原点，x正东、y正北，单位米。干扰源个数未知。
干扰源特征: 每个干扰源频道互不相同且不变，频道编号1..20(共20个频道)；不同干扰源信号互不影响；
  分全向(360°)与定向(定向方向两侧各90°，共180°含边界，范围外无辐射)；
  覆盖角内信号均匀辐射，场强仅随距离衰减。
测向机: 返回"示向度"=最大场强方向方位角(x正向逆时针，[0,360))，误差范围[-1°,+1°]；
  同一地点电磁环境固定→重复检测不改变误差(误差不可通过原地重测平均消除)；
  灵敏度决定"有效接收半径"，各干扰源该半径在1000~1500m之间且不返回；
  不能同时检测多频道，切换频道耗时1秒；移动中无法检测，须停下；
  调整天线获得稳定读数(或确认无信号)耗时5秒；机器狗沿直线行进，速度5m/s；
  光学探测仪在距干扰源<=20m可精确定位并激光清除；精确定位3秒，清除2秒(合计成功5秒，未发现3秒)；
  检测点距干扰源<=5m且在覆盖角内时信号过强无法获示向度(返回near)，可直接清除。
问题1: 已知若干检测点坐标及某干扰源关于这些检测点的示向度，给出用交会定位法计算
  多边形定位区域"直径"(区域内任意两点距离最大值)的算法。
  并回答: 以定位区域直径为直径的圆能否覆盖此定位区域?
问题2: 已知某全向干扰源在一个检测点处测得的示向度，给出第二个检测点的选择策略，
  以获得较好定位效果，并给出第二个检测点的候选区域。
问题3: 干扰源均为全向，总数10-16个但具体未知。机器狗从原点出发，测向机初始频道1，
  自动定位并清除。确保全部清除前提下，完成时间越短越好。制定搜索定位清除策略与算法。
  用模拟器演练测试(不限次数)，再做3次正式测试。统计: 被清除比例、平均定位清除时间。
  平均定位清除时间 = 定位清除总时间 / 被清除个数（总时间含移动+切频道+检测+光学定位+清除）
问题4: 既有全向又有定向干扰源，总数仍10-16，定向个数与定向方向均未知，其余同问题3。
模拟器: HTTP+JSON，4条指令 /enter /measure /clear /exit，默认 http://127.0.0.1:2026 仅监听回环。
  /measure 返回 no_signal | near | direction(+svd_deg，两位小数)。
  移动与切频道无独立指令，由 position/channel 参数变化推断。
  虚拟时间上限360000s(100h)；程序运行现实时间上限1200s(20min)，25分钟窗口取先到者。
  Windows绿色软件，需联网登录(参赛队号+队员1姓名+手机号)，同账号不可双设备。
  正式测试各3次机会，2026-09-13 17:30后不能启动新测试。""",

    "C": """CUMCM2026 C题 微网与外部电网电力调控策略
背景: 微网含光伏+储能+小区负载，可向外网购电。低价时段/光伏有余时充电，高价时段放电。
目标: 用负载、光伏发电量、储能当前储电量，给出购电量与储能充放电量，
  既满足小区负载，又尽可能节省购电费用。
储能参数(附录1): 最大容量12000 kWh，最大充放电功率5000 kW，
  2025-01-01 00:00 电量6000 kWh，运行中电量须保持1200~10800 kWh，充放电效率90%。
附件1: 某天不同时间的电价(元/kWh)、小区负载(kW)、光伏发电预测功率(kW)，10分钟粒度144点。
  实测: 电价 0.3713~1.3952(峰谷比3.758)；负载 3309~5959 kW；光伏预测峰值7612 kW；
        全天负载电量111025 kWh，光伏55483 kWh；净负载为负(光伏过剩)的时段30/144。
附件2: 2025.1.1-12.31 每天10分钟粒度的小区负载(kW)与光伏发电实际功率(kW)，各365×144。
  实测: 光伏日电量 34361/55483/75223 kWh (min/mean/max)；负载日电量均值111025 kWh；
        光伏装机峰值10216 kW。
附件3: 2025.1.1-12.31 每天 0:00/6:00/12:00/18:00 发布的未来24小时"整点"光伏功率预报(kW)，1460行。
  实测预报精度(对比附件2实际整点值): 0:00预报 MAE 181.3 kW / RMSE 370.6；
        白天7-20h 0:00预报 MAE 309.6 kW(占实际均值3933的7.9%)；
        6:00预报覆盖7-24h MAE 126.0 (同期0:00预报240.8)；
        12:00预报覆盖13-24h MAE 34.2 (同期0:00预报186.7)。→ 预报误差随时效显著衰减。
附件4: 2025.1.1-12.31 每天10分钟粒度电价(元/kWh)，波动电价。
  实测: 0.0076~1.7936，日均峰谷比5.74。
问题1: 每天电价与负载相同，给定光伏一天预测数据；微网供电不低于负载；储能0:00与24:00储电量相同。
  在每天0:00制定当天计划购电策略的数学模型。用附件1+附录1给出具体策略，
  指定时段购电量、全天购电量与购电费、储能指定时段充放电量及0:00/24:00储电量，存result1.xlsx。
问题2: 电价相同但负载与光伏随时间变化；供电低于负载须紧急购电，电价为交易时刻电价的5倍。
  除紧急购电外其他时段按计划购电量计费。每天0:00制定当天计划购电策略。
  用附件1电价+附件2数据。指定日期(2025.3.20/6.21/9.23/12.21)结果+紧急购电，存result2.xlsx。
  result2 覆盖 2025.2.1-12.31（334天）。
问题3: 每天0:00/6:00/12:00/18:00可获未来24小时整点光伏预报。可按0:00预报定计划，
  也可按其他时刻预报调整。计划购电量高于调整购电量的部分，违约电价为交易时刻电价的50%；
  调整购电量高于计划购电量的部分，超出部分电价为交易时刻电价的1.5倍。
  总费用=计划购电费+紧急购电费+调整购电相关费用。建模并用附件1电价+附件2+附件3计算。
  须分析"是否需要引入其他时刻的预报制定调整购电策略"。存result3.xlsx。
问题4: 外网电价实时波动。针对波动电价建模，用附件2+附件3+附件4重新计算问题2和问题3。
  存 result4-2.xlsx(对应问题2) 与 result4-3.xlsx(对应问题3)。""",
}

ROLES = {
    "modeler": """你是资深数学建模竞赛(CUMCM)国奖级选手兼指导教师。针对下面这道题，给出你独立的完整建模路线。
必须包含:
1) 每一问的数学模型: 决策变量/状态变量、目标函数、约束、方程(写出具体公式，不要只说方法名)
2) 求解算法与关键数值细节(离散格式、步长、稳定性条件、求解器选择)
3) 每一问最容易做错的地方(具体到公式项或边界条件)
4) 你认为本题的"拉分点"在哪一问、什么做法能拿高分
5) 预估各问实现工作量(小时)
要求: 用中文。写具体公式和数字，不要泛泛而谈方法论。不确定的地方明确标注"不确定"。""",

    "redteam": """你是数学建模竞赛的红队(Red Team)评审，任务是找出这道题里"最容易让队伍翻车"的陷阱和常见错误答案。
必须包含:
1) 题面中的隐藏陷阱/歧义点: 逐条列出，说明为什么容易误读、错误理解会导致什么后果
2) 大多数队伍会给出的"看似正确实则错误"的答案，以及正确答案应该是什么
3) 数据本身的坑(单位、粒度、边界、缺失、自相矛盾之处)
4) 数值方法上的失稳/发散风险点，以及触发条件
5) 如果你是评委，你会问哪3个问题让大部分队伍答不上来
要求: 用中文。每条给出具体的判断依据。只写你真正认为是陷阱的，不要为了凑数编造。""",

    "critic": """你是CUMCM全国评阅专家。请对这道题做"评分视角"分析。
必须包含:
1) 这道题的难度分布: 哪一问是分水岭，哪一问大部分队伍能做对
2) 区分度最高的环节是什么(具体到某一问的某个技术点)
3) 一份"能拿国一"的答卷和"只能拿省奖"的答卷，具体差别在哪
4) 本题最常见的"做了大量工作但不得分"的情况
5) 你对选这道题的建议: 适合什么类型的队伍，不适合什么类型
要求: 用中文。基于题目本身的技术结构判断，不要套用泛泛的评奖经验。""",
}

# 跨家族分配：每个角色用不同家族，避免同家族互评
ASSIGN = [
    ("A", "modeler",  "claude-opus-5"),
    ("A", "modeler",  "gpt-5.6-terra"),
    ("A", "redteam",  "glm-5"),
    ("A", "critic",   "deepseek-3.2"),
    ("B", "modeler",  "claude-opus-5"),
    ("B", "modeler",  "gpt-5.6-terra"),
    ("B", "redteam",  "glm-5"),
    ("B", "critic",   "minimax-m2.5"),
    ("C", "modeler",  "claude-opus-5"),
    ("C", "modeler",  "gpt-5.6-terra"),
    ("C", "redteam",  "glm-5"),
    ("C", "critic",   "deepseek-3.2"),
]


NO_TOOL = (
    "\n\n【重要执行约束】\n"
    "你没有任何可用工具，无法联网搜索，也无法调用函数。禁止输出任何工具调用、函数调用、"
    "搜索请求或 <function_calls> 之类的标记。禁止说“我先搜索一下”“让我查一下”。\n"
    "题目所需的全部信息已在用户消息中给出，直接基于这些信息和你自己的专业知识作答。\n"
    "立即开始输出正式的分析内容，不要有任何前置说明。"
)


def one(prob, role, model):
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ROLES[role] + NO_TOOL},
                {"role": "user", "content": PROBLEMS[prob]},
            ],
            max_tokens=8000,
        )
        txt = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        return {
            "problem": prob, "role": role, "model": model, "ok": True,
            "elapsed_s": round(time.time() - t0, 1),
            "tokens": (usage.total_tokens if usage else None),
            "content": txt,
        }
    except Exception as e:
        return {"problem": prob, "role": role, "model": model, "ok": False,
                "elapsed_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}


BAD_MARKS = ("function_calls", "DSML", "invoke name=", "<tool", "我先搜索", "让我查",
             "我来先搜索", "先搜索一些")
MIN_LEN = 1500


def one_robust(prob, role, model, attempts=3):
    """空回复/工具标记/过短 -> 自动重试，重试时追加更强约束。"""
    last = None
    for k in range(attempts):
        res = one(prob, role, model)
        last = res
        if not res["ok"]:
            time.sleep(3)
            continue
        txt = res.get("content", "") or ""
        too_short = len(txt) < MIN_LEN
        has_bad = any(m in txt for m in BAD_MARKS)
        if not too_short and not has_bad:
            res["attempts"] = k + 1
            return res
        reason = "过短" if too_short else "含工具标记"
        print(f"    [retry {k+1}] {prob}/{role}/{model} {reason} len={len(txt)}", flush=True)
        time.sleep(2)
    if last is None:
        return {"problem": prob, "role": role, "model": model, "ok": False,
                "elapsed_s": 0.0, "attempts": attempts,
                "error": "no response after retries"}
    if last.get("ok"):
        last["degraded"] = True
    last["attempts"] = attempts
    return last


def main():
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one_robust, p, r, m): (p, r, m) for p, r, m in ASSIGN}
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            tag = "OK " if res["ok"] else "ERR"
            if res.get("degraded"):
                tag = "DEG"
            print(f"[{tag}] {res['problem']}/{res['role']}/{res['model']} "
                  f"{res['elapsed_s']}s "
                  f"{len(res.get('content','')) if res['ok'] else res.get('error','')}",
                  flush=True)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    ok = sum(1 for r in results if r["ok"])
    print(f"\n完成 {ok}/{len(ASSIGN)}  -> {OUT}")


if __name__ == "__main__":
    main()
