from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSERVED_PATH = PROJECT_ROOT / "data" / "processed" / "volume_observed_20min.csv"
SUBMISSION_PATH = PROJECT_ROOT / "data" / "submission" / "submission_phase1.csv"
OUTPUT_PATH = PROJECT_ROOT / "visualization" / "traffic_flow_dashboard.html"

VALID_COMBOS = {(1, 0), (1, 1), (2, 0), (3, 0), (3, 1)}
DIRECTION_LABELS = {0: "Entry", 1: "Exit"}
SOURCE_LABELS = {"observed": "历史观测", "prediction": "预测结果"}
WINDOW_RE = re.compile(r"\[(.*?),")


def parse_window_start(row: dict[str, str]) -> datetime:
    if row.get("time_window_start"):
        return datetime.fromisoformat(row["time_window_start"])
    match = WINDOW_RE.search(row["time_window"])
    if not match:
        raise ValueError(f"无法解析 time_window: {row.get('time_window')}")
    return datetime.fromisoformat(match.group(1))


def make_time_window(start: datetime) -> str:
    end = start + timedelta(minutes=20)
    return f"[{start:%Y-%m-%d %H:%M:%S},{end:%Y-%m-%d %H:%M:%S})"


def load_volume_rows(path: Path, source: str) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件: {path.relative_to(PROJECT_ROOT)}")

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                tollgate_id = int(float(row["tollgate_id"]))
                direction = int(float(row["direction"]))
                volume = float(row["volume"])
                start = parse_window_start(row)
            except (KeyError, TypeError, ValueError):
                continue

            if (tollgate_id, direction) not in VALID_COMBOS:
                continue

            rows.append(
                {
                    "time": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": start.strftime("%Y-%m-%d"),
                    "timeLabel": start.strftime("%H:%M"),
                    "hour": start.hour,
                    "minute": start.minute,
                    "timeWindow": row.get("time_window") or make_time_window(start),
                    "tollgate": tollgate_id,
                    "direction": direction,
                    "directionLabel": DIRECTION_LABELS[direction],
                    "combo": f"Tollgate {tollgate_id} - {DIRECTION_LABELS[direction]}",
                    "volume": round(volume, 4),
                    "source": source,
                    "sourceLabel": SOURCE_LABELS[source],
                }
            )
    return rows


def build_html(data: list[dict[str, object]]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>高速收费站车流量可视化</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d9e1ec;
      --entry: #36a9e1;
      --exit: #ef6471;
      --map-bg: #30343d;
      --yellow: #f4d774;
      --shadow: 0 14px 36px rgba(15, 23, 42, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Microsoft YaHei", sans-serif;
    }}
    header {{
      padding: 26px 34px 18px;
      background: linear-gradient(135deg, #172033, #2f415f);
      color: #fff;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 750;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: #dbeafe;
      font-size: 15px;
    }}
    main {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 22px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .control, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .control {{
      padding: 12px;
      min-width: 0;
    }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }}
    select, input[type="range"], button {{
      width: 100%;
      font: inherit;
    }}
    select, button {{
      height: 38px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
    }}
    button {{
      cursor: pointer;
      background: #1f6feb;
      border-color: #1f6feb;
      color: #fff;
      font-weight: 650;
    }}
    .checks {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      min-height: 38px;
    }}
    .checks label {{
      margin: 0;
      color: var(--ink);
      font-size: 13px;
      white-space: nowrap;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .metric strong {{
      font-size: 24px;
      line-height: 1.15;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.32fr 0.68fr;
      gap: 16px;
      align-items: start;
    }}
    .card {{
      padding: 16px;
      overflow: hidden;
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .map-wrap {{
      background: var(--map-bg);
      border-radius: 8px;
      overflow: hidden;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .route {{
      fill: none;
      stroke-linecap: round;
      opacity: 0.92;
      transition: stroke-width 180ms ease, opacity 180ms ease;
    }}
    .route.dim {{ opacity: 0.18; }}
    .entry {{ stroke: var(--entry); }}
    .exit {{ stroke: var(--exit); }}
    .route-hit {{
      fill: none;
      stroke: transparent;
      stroke-width: 24;
      pointer-events: stroke;
    }}
    .arrow-entry {{ fill: var(--entry); }}
    .arrow-exit {{ fill: var(--exit); }}
    .intersection {{
      fill: var(--yellow);
      stroke: rgba(244, 215, 116, 0.8);
    }}
    .tollgate {{
      fill: rgba(216, 222, 229, 0.66);
      stroke: rgba(255, 255, 255, 0.16);
    }}
    .map-text {{
      fill: #e5e7eb;
      font-size: 16px;
      font-weight: 600;
    }}
    .node-label {{
      fill: var(--yellow);
      font-size: 14px;
      font-weight: 650;
    }}
    .flow-label rect {{
      fill: rgba(15, 23, 42, 0.76);
      stroke: rgba(255, 255, 255, 0.18);
      rx: 5;
    }}
    .flow-label text {{
      fill: #fff;
      font-size: 12px;
      font-weight: 650;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend i {{
      display: inline-block;
      width: 34px;
      height: 4px;
      border-radius: 99px;
      margin-right: 8px;
      vertical-align: middle;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 126px 1fr 58px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}
    .bar-track {{
      height: 18px;
      background: #edf2f7;
      border-radius: 99px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      min-width: 3px;
      border-radius: 99px;
    }}
    .trend {{
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #e5eaf2;
      padding: 8px 6px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: #f8fafc;
    }}
    .tooltip {{
      position: fixed;
      z-index: 10;
      display: none;
      padding: 8px 10px;
      background: rgba(15, 23, 42, 0.92);
      color: #fff;
      border-radius: 6px;
      font-size: 12px;
      pointer-events: none;
      max-width: 260px;
    }}
    .empty {{
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 1050px) {{
      .controls {{ grid-template-columns: repeat(2, 1fr); }}
      .summary {{ grid-template-columns: repeat(2, 1fr); }}
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>高速收费站车流量可视化</h1>
    <p>按 20 分钟窗口展示 Tollgate 1/2/3 与 Entry/Exit 方向组合流量，数据生成时间：{generated_at}</p>
  </header>
  <main>
    <section class="controls">
      <div class="control">
        <label for="sourceSelect">数据来源</label>
        <select id="sourceSelect">
          <option value="all">历史观测 + 预测结果</option>
          <option value="prediction">预测结果</option>
          <option value="observed">历史观测</option>
        </select>
      </div>
      <div class="control">
        <label for="dateSelect">日期</label>
        <select id="dateSelect"></select>
      </div>
      <div class="control">
        <label for="sessionSelect">时段</label>
        <select id="sessionSelect">
          <option value="all">全部</option>
          <option value="morning">上午 08:00-10:00</option>
          <option value="evening">下午 17:00-19:00</option>
          <option value="lead">先导窗口</option>
        </select>
      </div>
      <div class="control">
        <label for="timeSelect">20 分钟窗口</label>
        <select id="timeSelect"></select>
      </div>
      <div class="control">
        <label for="timeSlider">时间轴</label>
        <input id="timeSlider" type="range" min="0" value="0" step="1">
      </div>
      <div class="control">
        <label>播放</label>
        <button id="playBtn">播放时间窗口</button>
      </div>
    </section>

    <section class="controls" style="grid-template-columns: 1fr;">
      <div class="control">
        <label>收费站-方向组合</label>
        <div id="comboChecks" class="checks"></div>
      </div>
    </section>

    <section class="summary">
      <div class="metric"><span>当前窗口</span><strong id="metricWindow">-</strong></div>
      <div class="metric"><span>当前总流量</span><strong id="metricTotal">-</strong></div>
      <div class="metric"><span>最高组合</span><strong id="metricTopCombo">-</strong></div>
      <div class="metric"><span>最高流量</span><strong id="metricTopVolume">-</strong></div>
    </section>

    <section class="layout">
      <div class="card">
        <h2>收费站路网示意图</h2>
        <div class="map-wrap">
          <svg viewBox="0 0 1000 560" role="img" aria-label="收费站路网流量示意图">
            <defs>
              <marker id="arrowEntry" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path class="arrow-entry" d="M 0 0 L 10 5 L 0 10 z"></path>
              </marker>
              <marker id="arrowExit" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path class="arrow-exit" d="M 0 0 L 10 5 L 0 10 z"></path>
              </marker>
            </defs>

            <rect class="intersection" x="42" y="184" width="36" height="122"></rect>
            <rect class="intersection" x="918" y="170" width="36" height="160"></rect>
            <rect class="intersection" x="920" y="26" width="36" height="106"></rect>
            <text class="node-label" x="10" y="340">Intersection A</text>
            <text class="node-label" x="840" y="358">Intersection B</text>
            <text class="node-label" x="785" y="68">Intersection C</text>

            <circle class="tollgate" cx="330" cy="156" r="48"></circle>
            <circle class="tollgate" cx="314" cy="340" r="30"></circle>
            <circle class="tollgate" cx="610" cy="294" r="62"></circle>
            <text class="map-text" x="188" y="118">Tollgate 1</text>
            <text class="map-text" x="210" y="380">Tollgate 2</text>
            <text class="map-text" x="654" y="366">Tollgate3</text>
            <text class="map-text" x="36" y="80" style="font-size: 30px;">IN</text>

            <path id="route-1-0" class="route entry" marker-end="url(#arrowEntry)" d="M 922 76 C 880 160, 742 188, 650 206 S 444 204, 330 156"></path>
            <path id="route-1-1" class="route exit" marker-end="url(#arrowExit)" d="M 330 156 C 286 240, 164 246, 78 214"></path>
            <path id="route-2-0" class="route entry" marker-end="url(#arrowEntry)" d="M 78 254 C 158 258, 230 278, 314 340"></path>
            <path id="route-3-0" class="route entry" marker-end="url(#arrowEntry)" d="M 78 282 C 220 300, 424 330, 610 294"></path>
            <path id="route-3-1" class="route exit" marker-end="url(#arrowExit)" d="M 610 294 C 708 270, 812 258, 918 244"></path>

            <path data-route="1-0" class="route-hit" d="M 922 76 C 880 160, 742 188, 650 206 S 444 204, 330 156"></path>
            <path data-route="1-1" class="route-hit" d="M 330 156 C 286 240, 164 246, 78 214"></path>
            <path data-route="2-0" class="route-hit" d="M 78 254 C 158 258, 230 278, 314 340"></path>
            <path data-route="3-0" class="route-hit" d="M 78 282 C 220 300, 424 330, 610 294"></path>
            <path data-route="3-1" class="route-hit" d="M 610 294 C 708 270, 812 258, 918 244"></path>

            <g id="label-1-0" class="flow-label" transform="translate(420 130)"><rect width="120" height="42"></rect><text x="10" y="17">T1 Entry</text><text class="value" x="10" y="34">0</text></g>
            <g id="label-1-1" class="flow-label" transform="translate(188 172)"><rect width="120" height="42"></rect><text x="10" y="17">T1 Exit</text><text class="value" x="10" y="34">0</text></g>
            <g id="label-2-0" class="flow-label" transform="translate(350 356)"><rect width="120" height="42"></rect><text x="10" y="17">T2 Entry</text><text class="value" x="10" y="34">0</text></g>
            <g id="label-3-0" class="flow-label" transform="translate(684 274)"><rect width="120" height="42"></rect><text x="10" y="17">T3 Entry</text><text class="value" x="10" y="34">0</text></g>
            <g id="label-3-1" class="flow-label" transform="translate(684 328)"><rect width="120" height="42"></rect><text x="10" y="17">T3 Exit</text><text class="value" x="10" y="34">0</text></g>

            <line x1="34" y1="500" x2="86" y2="500" stroke="var(--entry)" stroke-width="5"></line>
            <line x1="34" y1="528" x2="86" y2="528" stroke="var(--exit)" stroke-width="5"></line>
            <text x="104" y="505" fill="var(--entry)" font-size="18">Highway Entry</text>
            <text x="104" y="533" fill="var(--exit)" font-size="18">Highway Exit</text>
          </svg>
        </div>
        <div class="legend">
          <span><i style="background: var(--entry);"></i>Entry</span>
          <span><i style="background: var(--exit);"></i>Exit</span>
          <span>线条越粗表示当前窗口车流量越高</span>
        </div>
      </div>

      <div class="card">
        <h2>当前窗口组合流量</h2>
        <div id="barChart"></div>
        <h2 style="margin-top: 22px;">当前窗口数据</h2>
        <div id="tableWrap"></div>
      </div>
    </section>

    <section class="card trend">
      <h2>选中日期内流量趋势</h2>
      <svg id="trendChart" viewBox="0 0 1200 340"></svg>
    </section>
  </main>
  <div id="tooltip" class="tooltip"></div>

  <script>
    const TRAFFIC_DATA = {data_json};
    const COMBOS = [
      {{ key: "1-0", label: "Tollgate 1 - Entry", tollgate: 1, direction: 0, color: "#36a9e1" }},
      {{ key: "1-1", label: "Tollgate 1 - Exit", tollgate: 1, direction: 1, color: "#ef6471" }},
      {{ key: "2-0", label: "Tollgate 2 - Entry", tollgate: 2, direction: 0, color: "#36a9e1" }},
      {{ key: "3-0", label: "Tollgate 3 - Entry", tollgate: 3, direction: 0, color: "#36a9e1" }},
      {{ key: "3-1", label: "Tollgate 3 - Exit", tollgate: 3, direction: 1, color: "#ef6471" }}
    ];
    const SOURCE_ORDER = {{ observed: 0, prediction: 1 }};
    const globalMax = Math.max(...TRAFFIC_DATA.map(d => d.volume), 1);
    let playing = false;
    let timer = null;

    const sourceSelect = document.getElementById("sourceSelect");
    const dateSelect = document.getElementById("dateSelect");
    const sessionSelect = document.getElementById("sessionSelect");
    const timeSelect = document.getElementById("timeSelect");
    const timeSlider = document.getElementById("timeSlider");
    const playBtn = document.getElementById("playBtn");
    const comboChecks = document.getElementById("comboChecks");
    const tooltip = document.getElementById("tooltip");

    function formatVolume(value) {{
      return Number(value || 0).toFixed(1);
    }}

    function comboKey(row) {{
      return `${{row.tollgate}}-${{row.direction}}`;
    }}

    function sourceFiltered() {{
      const source = sourceSelect.value;
      return TRAFFIC_DATA.filter(row => source === "all" || row.source === source);
    }}

    function sessionFiltered(rows) {{
      const session = sessionSelect.value;
      if (session === "morning") return rows.filter(row => row.hour === 8 || row.hour === 9);
      if (session === "evening") return rows.filter(row => row.hour === 17 || row.hour === 18);
      if (session === "lead") return rows.filter(row => [6, 7, 15, 16].includes(row.hour));
      return rows;
    }}

    function selectedCombos() {{
      return Array.from(document.querySelectorAll("[name='combo']:checked")).map(input => input.value);
    }}

    function uniqueSorted(rows, key) {{
      return Array.from(new Set(rows.map(row => row[key]))).sort();
    }}

    function rebuildDates() {{
      const dates = uniqueSorted(sourceFiltered(), "date");
      const previous = dateSelect.value;
      dateSelect.innerHTML = dates.map(date => `<option value="${{date}}">${{date}}</option>`).join("");
      const predictionFirst = uniqueSorted(TRAFFIC_DATA.filter(row => row.source === "prediction"), "date")[0];
      dateSelect.value = dates.includes(previous) ? previous : (dates.includes(predictionFirst) ? predictionFirst : dates[0]);
    }}

    function rebuildTimes() {{
      const base = sessionFiltered(sourceFiltered().filter(row => row.date === dateSelect.value));
      const times = uniqueSorted(base, "timeLabel");
      const previous = timeSelect.value;
      timeSelect.innerHTML = times.map(time => `<option value="${{time}}">${{time}}</option>`).join("");
      timeSelect.value = times.includes(previous) ? previous : times[0];
      timeSlider.max = Math.max(times.length - 1, 0);
      timeSlider.value = Math.max(times.indexOf(timeSelect.value), 0);
    }}

    function rebuildCombos() {{
      comboChecks.innerHTML = COMBOS.map(combo => `
        <label><input type="checkbox" name="combo" value="${{combo.key}}" checked> ${{combo.label}}</label>
      `).join("");
      comboChecks.querySelectorAll("input").forEach(input => input.addEventListener("change", update));
    }}

    function currentRows() {{
      const wanted = new Set(selectedCombos());
      const rows = sessionFiltered(sourceFiltered()).filter(row =>
        row.date === dateSelect.value &&
        row.timeLabel === timeSelect.value &&
        wanted.has(comboKey(row))
      );
      const best = new Map();
      rows.sort((a, b) => SOURCE_ORDER[a.source] - SOURCE_ORDER[b.source])
        .forEach(row => best.set(comboKey(row), row));
      return Array.from(best.values()).sort((a, b) => a.tollgate - b.tollgate || a.direction - b.direction);
    }}

    function dayRows() {{
      const wanted = new Set(selectedCombos());
      return sessionFiltered(sourceFiltered()).filter(row =>
        row.date === dateSelect.value && wanted.has(comboKey(row))
      );
    }}

    function updateMetrics(rows) {{
      const total = rows.reduce((sum, row) => sum + row.volume, 0);
      const top = [...rows].sort((a, b) => b.volume - a.volume)[0];
      document.getElementById("metricWindow").textContent = rows[0]?.timeLabel || "-";
      document.getElementById("metricTotal").textContent = formatVolume(total);
      document.getElementById("metricTopCombo").textContent = top ? top.combo : "-";
      document.getElementById("metricTopVolume").textContent = top ? formatVolume(top.volume) : "-";
    }}

    function updateMap(rows) {{
      const byCombo = new Map(rows.map(row => [comboKey(row), row]));
      COMBOS.forEach(combo => {{
        const row = byCombo.get(combo.key);
        const volume = row ? row.volume : 0;
        const path = document.getElementById(`route-${{combo.key}}`);
        const label = document.querySelector(`#label-${{combo.key}} .value`);
        const group = document.getElementById(`label-${{combo.key}}`);
        const width = volume > 0 ? 3 + 13 * volume / globalMax : 2;
        path.setAttribute("stroke-width", width.toFixed(2));
        path.classList.toggle("dim", !row);
        group.style.opacity = row ? 1 : 0.28;
        label.textContent = formatVolume(volume);
      }});
    }}

    function updateBars(rows) {{
      const max = Math.max(...rows.map(row => row.volume), 1);
      document.getElementById("barChart").innerHTML = rows.length ? rows.map(row => {{
        const color = row.direction === 0 ? "var(--entry)" : "var(--exit)";
        const width = Math.max(3, row.volume / max * 100);
        return `
          <div class="bar-row">
            <div>${{row.combo.replace("Tollgate ", "T")}}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${{width}}%; background:${{color}};"></div></div>
            <strong>${{formatVolume(row.volume)}}</strong>
          </div>`;
      }}).join("") : `<div class="empty">当前窗口没有数据</div>`;
    }}

    function updateTable(rows) {{
      document.getElementById("tableWrap").innerHTML = rows.length ? `
        <table>
          <thead><tr><th>来源</th><th>组合</th><th>窗口</th><th>流量</th></tr></thead>
          <tbody>
            ${{rows.map(row => `<tr><td>${{row.sourceLabel}}</td><td>${{row.combo}}</td><td>${{row.timeLabel}}</td><td>${{formatVolume(row.volume)}}</td></tr>`).join("")}}
          </tbody>
        </table>` : `<div class="empty">当前筛选无数据</div>`;
    }}

    function updateTrend(rows) {{
      const svg = document.getElementById("trendChart");
      const width = 1200, height = 340;
      const pad = {{ left: 58, right: 20, top: 18, bottom: 44 }};
      svg.innerHTML = "";
      if (!rows.length) {{
        svg.innerHTML = `<text x="600" y="170" text-anchor="middle" fill="#64748b">当前日期没有趋势数据</text>`;
        return;
      }}
      const times = uniqueSorted(rows, "timeLabel");
      const max = Math.max(...rows.map(row => row.volume), 1);
      const x = time => pad.left + (times.indexOf(time) / Math.max(times.length - 1, 1)) * (width - pad.left - pad.right);
      const y = value => height - pad.bottom - (value / max) * (height - pad.top - pad.bottom);

      const axis = document.createElementNS("http://www.w3.org/2000/svg", "g");
      axis.innerHTML = `
        <line x1="${{pad.left}}" y1="${{height-pad.bottom}}" x2="${{width-pad.right}}" y2="${{height-pad.bottom}}" stroke="#cbd5e1"/>
        <line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height-pad.bottom}}" stroke="#cbd5e1"/>
        <text x="${{pad.left-8}}" y="${{pad.top+8}}" text-anchor="end" fill="#64748b" font-size="12">${{formatVolume(max)}}</text>
        <text x="${{pad.left-8}}" y="${{height-pad.bottom}}" text-anchor="end" fill="#64748b" font-size="12">0</text>`;
      svg.appendChild(axis);

      times.forEach((time, idx) => {{
        if (idx % Math.ceil(times.length / 8) !== 0 && idx !== times.length - 1) return;
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x(time));
        text.setAttribute("y", height - 16);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", "#64748b");
        text.setAttribute("font-size", "12");
        text.textContent = time;
        svg.appendChild(text);
      }});

      COMBOS.forEach(combo => {{
        const comboRows = rows.filter(row => comboKey(row) === combo.key).sort((a, b) => a.time.localeCompare(b.time));
        if (!comboRows.length) return;
        const points = comboRows.map(row => `${{x(row.timeLabel).toFixed(1)}},${{y(row.volume).toFixed(1)}}`).join(" ");
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyline.setAttribute("points", points);
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", combo.color);
        polyline.setAttribute("stroke-width", "2.5");
        polyline.setAttribute("stroke-linejoin", "round");
        polyline.setAttribute("stroke-linecap", "round");
        svg.appendChild(polyline);
        comboRows.forEach(row => {{
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x(row.timeLabel));
          circle.setAttribute("cy", y(row.volume));
          circle.setAttribute("r", "3.5");
          circle.setAttribute("fill", combo.color);
          circle.dataset.tip = `${{row.combo}}<br>${{row.timeLabel}}: ${{formatVolume(row.volume)}}`;
          circle.addEventListener("mousemove", showTip);
          circle.addEventListener("mouseleave", hideTip);
          svg.appendChild(circle);
        }});
      }});
    }}

    function showTip(event) {{
      tooltip.innerHTML = event.target.dataset.tip || "";
      tooltip.style.display = "block";
      tooltip.style.left = `${{event.clientX + 12}}px`;
      tooltip.style.top = `${{event.clientY + 12}}px`;
    }}

    function hideTip() {{
      tooltip.style.display = "none";
    }}

    function update() {{
      const rows = currentRows();
      updateMetrics(rows);
      updateMap(rows);
      updateBars(rows);
      updateTable(rows);
      updateTrend(dayRows());
    }}

    function resetForSourceOrSession() {{
      rebuildDates();
      rebuildTimes();
      update();
    }}

    sourceSelect.addEventListener("change", resetForSourceOrSession);
    dateSelect.addEventListener("change", () => {{ rebuildTimes(); update(); }});
    sessionSelect.addEventListener("change", () => {{ rebuildTimes(); update(); }});
    timeSelect.addEventListener("change", () => {{
      const options = Array.from(timeSelect.options).map(option => option.value);
      timeSlider.value = Math.max(options.indexOf(timeSelect.value), 0);
      update();
    }});
    timeSlider.addEventListener("input", () => {{
      const idx = Number(timeSlider.value);
      if (timeSelect.options[idx]) timeSelect.value = timeSelect.options[idx].value;
      update();
    }});
    playBtn.addEventListener("click", () => {{
      playing = !playing;
      playBtn.textContent = playing ? "暂停播放" : "播放时间窗口";
      if (playing) {{
        timer = setInterval(() => {{
          const next = (Number(timeSlider.value) + 1) % Math.max(timeSelect.options.length, 1);
          timeSlider.value = next;
          if (timeSelect.options[next]) timeSelect.value = timeSelect.options[next].value;
          update();
        }}, 900);
      }} else {{
        clearInterval(timer);
      }}
    }});
    document.querySelectorAll(".route-hit").forEach(hit => {{
      hit.addEventListener("mousemove", event => {{
        const row = currentRows().find(item => comboKey(item) === hit.dataset.route);
        hit.dataset.tip = row ? `${{row.combo}}<br>${{row.timeWindow}}<br>车流量：${{formatVolume(row.volume)}}` : "当前筛选无该组合数据";
        showTip(event);
      }});
      hit.addEventListener("mouseleave", hideTip);
    }});

    rebuildCombos();
    rebuildDates();
    rebuildTimes();
    update();
  </script>
</body>
</html>
"""


def main() -> None:
    data = load_volume_rows(OBSERVED_PATH, "observed") + load_volume_rows(
        SUBMISSION_PATH, "prediction"
    )
    data.sort(key=lambda row: (row["time"], row["tollgate"], row["direction"], row["source"]))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_html(data), encoding="utf-8")
    print(f"HTML visualization written: {OUTPUT_PATH}")
    print(f"Rows embedded: {len(data)}")


if __name__ == "__main__":
    main()
