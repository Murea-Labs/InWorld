/* InWorld project page interactions. No dependencies besides KaTeX (vendored). */
(function () {
  "use strict";
  var D = window.INWORLD_DATA;
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  function el(tag, attrs, children) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "text") n.textContent = attrs[k];
      else if (k === "class") n.className = attrs[k];
      else if (k === "style") n.setAttribute("style", attrs[k]);
      else n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach(function (c) { if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function fmt(v, d) { return v == null ? "—" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d); }
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- math ---------- */
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
      throwOnError: false
    });
  }

  /* ---------- nav ---------- */
  var nav = $("#nav"), hero = $(".hero");
  function navState() { nav.classList.toggle("is-solid", window.scrollY > hero.offsetHeight - 72); }
  var ticking = false;
  window.addEventListener("scroll", function () {
    if (!ticking) { ticking = true; requestAnimationFrame(function () { navState(); ticking = false; }); }
  }, { passive: true });
  window.addEventListener("resize", navState);
  navState();

  var links = $$(".nav__links a");
  if ("IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && byId[e.target.id]) {
          links.forEach(function (a) { a.removeAttribute("aria-current"); });
          byId[e.target.id].setAttribute("aria-current", "true");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  /* ---------- paper link: fall back to a disabled button until the PDF is uploaded ---------- */
  var paperBtn = $('.hero__actions a[href$=".pdf"]');
  if (paperBtn && location.protocol.indexOf("http") === 0) {
    fetch(paperBtn.getAttribute("href"), { method: "HEAD" }).then(function (r) {
      if (!r.ok) throw 0;
    }).catch(function () {
      paperBtn.removeAttribute("href");
      paperBtn.setAttribute("aria-disabled", "true");
      paperBtn.lastChild.textContent = " Paper, coming soon";
    });
  }

  /* ---------- the turn loop ---------- */
  (function loop() {
    var tabs = $$(".loop__node"), panels = tabs.map(function (t) { return document.getElementById(t.getAttribute("aria-controls")); });
    var arc = $("#loop-arc"), arrows = $("#loop-arrows"), prev = $("#loop-prev"), next = $("#loop-next");
    if (!tabs.length) return;
    var R = 46, C = 2 * Math.PI * R, cur = 0, angle = 180;
    arc.style.strokeDasharray = (C / 4 - 9) + " " + C;
    arc.style.transformBox = "view-box";
    arc.style.transformOrigin = "50% 50%";
    [315, 45, 135, 225].forEach(function (deg) {
      var t = deg * Math.PI / 180, px = 50 + R * Math.cos(t), py = 50 + R * Math.sin(t);
      var tx = -Math.sin(t), ty = Math.cos(t), nx = Math.cos(t), ny = Math.sin(t);
      var p = [[px + tx * 1.9, py + ty * 1.9], [px - tx * 1.3 + nx * 1.7, py - ty * 1.3 + ny * 1.7], [px - tx * 1.3 - nx * 1.7, py - ty * 1.3 - ny * 1.7]];
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", "M" + p.map(function (q) { return q[0].toFixed(2) + " " + q[1].toFixed(2); }).join("L") + "Z");
      path.setAttribute("class", "loop__arrow");
      arrows.appendChild(path);
    });
    function show(i, focus) {
      cur = (i + 4) % 4;
      tabs.forEach(function (t, k) {
        var on = k === cur;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        panels[k].hidden = !on;
      });
      if (focus) tabs[cur].focus();
      // the highlighted quarter is the stretch of the loop that leads into the selected operator
      var target = 180 + 90 * cur + 4.5;
      while (target - angle > 180) target -= 360;
      while (angle - target > 180) target += 360;
      angle = target;
      arc.style.transform = "rotate(" + angle + "deg)";
      prev.disabled = false; next.disabled = false;
      prev.textContent = cur === 0 ? "Previous turn" : "Previous step";
      next.textContent = cur === 3 ? "Back to planning" : "Next step";
    }
    tabs.forEach(function (t, k) {
      t.addEventListener("click", function () { show(k); });
      t.addEventListener("keydown", function (e) {
        var m = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
        if (m[e.key]) { e.preventDefault(); show(cur + m[e.key], true); }
        else if (e.key === "Home") { e.preventDefault(); show(0, true); }
        else if (e.key === "End") { e.preventDefault(); show(3, true); }
      });
    });
    prev.addEventListener("click", function () { show(cur - 1); });
    next.addEventListener("click", function () { show(cur + 1); });
    show(0);
  })();

  /* ---------- tooltip ---------- */
  var tip = $("#tooltip");
  function tipShow(e, value, lines) {
    tip.textContent = "";
    tip.appendChild(el("strong", { text: value }));
    lines.forEach(function (l) { if (l) tip.appendChild(el("div", { text: l })); });
    tip.classList.add("is-on");
    tipMove(e);
  }
  function tipMove(e) {
    var r = tip.getBoundingClientRect(), x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function tipHide() { tip.classList.remove("is-on"); }

  /* ---------- helpers ---------- */
  function niceStep(span) {
    var raw = span / 5, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function seg(options, current, onChange, label) {
    var wrap = el("div", { class: "seg", role: "group", "aria-label": label });
    options.forEach(function (o) {
      var b = el("button", { type: "button", "aria-pressed": o.key === current ? "true" : "false", text: o.label });
      b.addEventListener("click", function () {
        $$("button", wrap).forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onChange(o.key);
      });
      wrap.appendChild(b);
    });
    return wrap;
  }
  function legend(oursLabel) {
    return el("p", { class: "viz__legend" }, [el("span", { class: "is-ours", text: oursLabel }), el("span", { text: "Other systems" })]);
  }

  /* ---------- dot plot (one row per system, re-sorted per metric) ---------- */
  function DotPlot(host, items) {
    // items: [{name, ours, sub, meta}]
    var ROW = 32;
    var root = el("div", { class: "dots", role: "img" });
    var axis = el("div", { class: "dots__axis", "aria-hidden": "true" }, [el("span"), el("div", { class: "dots__ticks" }), el("span")]);
    var body = el("div", { class: "dots__body" });
    var overlay = el("div", { class: "dots__overlay", "aria-hidden": "true" }, [el("span"), el("div", { class: "g" }), el("span")]);
    body.appendChild(overlay);
    var rows = items.map(function (it) {
      var dot = el("span", { class: "dot" }), hit = el("span", { class: "hit" }), v = el("span", { class: "v" });
      var nm = el("span", { class: "nm" }, [it.name, it.sub ? el("small", { text: it.sub }) : null]);
      var row = el("div", { class: "dots__row" + (it.ours ? " is-ours" : ""), "aria-hidden": "true" }, [nm, el("span", { class: "tr" }, [dot, hit]), v]);
      body.appendChild(row);
      return { it: it, row: row, dot: dot, hit: hit, v: v, val: null };
    });
    root.appendChild(axis); root.appendChild(body); host.appendChild(root);
    var foot = el("p", { class: "viz__foot" }); host.appendChild(foot);

    rows.forEach(function (r) {
      r.row.addEventListener("pointerenter", function (e) {
        if (r.val == null) return;
        tipShow(e, r.vText, [r.it.name, r.rankText, r.it.meta]);
      });
      r.row.addEventListener("pointermove", tipMove);
      r.row.addEventListener("pointerleave", tipHide);
    });

    this.update = function (getVal, opt) {
      // opt: {decimals, higher, include(it) -> bool, title}
      var shown = rows.filter(function (r) { r.val = getVal(r.it); return opt.include(r.it); });
      var missing = shown.filter(function (r) { return r.val == null; });
      var present = shown.filter(function (r) { return r.val != null; });
      present.sort(function (a, b) { return opt.higher ? b.val - a.val : a.val - b.val; });
      var vals = present.map(function (r) { return r.val; });
      var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
      var step = niceStep(Math.max(mx - mn, 1e-6));
      var lo = Math.floor(mn / step) * step, hi = Math.ceil(mx / step) * step;
      if (lo === mn) lo -= step; if (hi === mx) hi += step;
      if (opt.zero && lo < 0 && mn >= 0) lo = 0;
      var span = hi - lo, pos = function (v) { return ((v - lo) / span) * 100; };

      var ticks = $(".dots__ticks", axis), g = $(".g", overlay);
      ticks.textContent = ""; g.textContent = "";
      var dec = step < 0.1 ? 2 : step < 1 ? 1 : 0;
      for (var t = lo; t <= hi + step / 2; t += step) {
        ticks.appendChild(el("span", { style: "left:" + pos(t) + "%", text: fmt(t, dec) }));
        g.appendChild(el("i", { style: "left:" + pos(t) + "%" }));
      }
      // rank among distinct values, ties share a rank
      var distinct = [];
      present.forEach(function (r) { if (distinct.indexOf(r.val) < 0) distinct.push(r.val); });
      present.forEach(function (r, i) {
        r.vText = fmt(r.val, opt.decimals);
        r.rankText = opt.ranked === false ? "Not ranked" : "Rank " + (distinct.indexOf(r.val) + 1) + " of " + present.length;
        r.row.classList.remove("is-missing");
        r.row.style.transform = "translateY(" + i * ROW + "px)";
        r.dot.style.left = r.hit.style.left = pos(r.val) + "%";
        r.v.textContent = r.vText;
      });
      rows.forEach(function (r) {
        if (present.indexOf(r) < 0) { r.row.classList.add("is-missing"); r.row.style.transform = "translateY(" + present.length * ROW + "px)"; }
      });
      body.style.height = present.length * ROW + "px";
      var ours = present.filter(function (r) { return r.it.ours; })[0];
      root.setAttribute("aria-label", opt.title + ". " + present.length + " systems" +
        (ours ? "; " + ours.it.name + " scores " + ours.vText + ", " + ours.rankText.toLowerCase() : "") +
        ". The full table is available below.");
      foot.textContent = (opt.note || "") + (missing.length ? (opt.note ? " " : "") + "No score reported for " + missing.map(function (r) { return r.it.name; }).join(", ") + "." : "");
      foot.hidden = !foot.textContent;
    };
  }

  /* ---------- data tables ---------- */
  function rankColumns(rows, cols) {
    // returns per-col {best, second} values among distinct values
    return cols.map(function (c) {
      if (!c.ranked) return null;
      var vs = [];
      rows.forEach(function (r) { var v = r[c.i]; if (v != null && vs.indexOf(v) < 0) vs.push(v); });
      vs.sort(function (a, b) { return c.higher ? b - a : a - b; });
      return { best: vs[0], second: vs[1] };
    });
  }
  function dataTable(caption, cols, groups, isOurs) {
    var all = []; groups.forEach(function (g) { all = all.concat(g.rows); });
    var ranks = rankColumns(all, cols);
    var thead = el("thead", null, [el("tr", null, cols.map(function (c, k) {
      return el("th", { scope: "col", title: c.title || "" , text: c.label + (c.ranked && c.higher === false ? " ↓" : "") });
    }))]);
    var tbody = el("tbody");
    groups.forEach(function (g) {
      if (g.label) tbody.appendChild(el("tr", { class: "group" }, [el("th", { scope: "rowgroup", colspan: cols.length, text: g.label })]));
      g.rows.forEach(function (r) {
        var tr = el("tr", { class: isOurs(r) ? "is-ours" : "" });
        cols.forEach(function (c, k) {
          var v = r[c.i], txt = typeof v === "number" ? fmt(v, c.d) : (v == null ? "—" : String(v));
          var cls = ""; var rk = ranks[k];
          if (rk && v === rk.best) cls = "b1"; else if (rk && v === rk.second) cls = "b2";
          tr.appendChild(k === 0 ? el("th", { scope: "row", text: txt }) : el("td", { class: cls, text: txt }));
        });
        tbody.appendChild(tr);
      });
    });
    return el("div", { class: "table-scroll" }, [el("table", { class: "data" }, [el("caption", { text: caption }), thead, tbody])]);
  }
  function tableToggle(label, table) {
    var d = el("details", { class: "table-toggle" }, [el("summary", { text: label }), table]);
    return d;
  }

  /* ---------- WBench ---------- */
  (function wbench() {
    var host = $("#wbench-viz"); if (!host) return;
    var W = D.wbench, dim = 0;
    var items = W.rows.map(function (r) { return { name: r[0], ours: !!r[7], vals: r.slice(1, 7) }; });
    var title = el("p", { class: "viz__title" }), sub = el("p", { class: "viz__sub", text: "0–100, higher is better. Baseline scores from a fixed WBench leaderboard snapshot." });
    var controls = el("div", { class: "controls" }, [seg(W.dims.map(function (d, i) { return { key: i, label: d.label }; }), 0, function (k) { dim = k; render(); }, "WBench dimension")]);
    host.appendChild(controls); host.appendChild(title); host.appendChild(sub); host.appendChild(legend("Murea-Pro"));
    var plot = new DotPlot(host, items);
    function render() {
      var name = W.dims[dim].label;
      title.textContent = "WBench Full-289, " + name;
      plot.update(function (it) { return it.vals[dim]; }, { decimals: 2, higher: true, include: function () { return true; }, title: "WBench Full-289 " + name + " scores" });
    }
    render();
    var cols = [{ i: 0, label: "Model" }].concat(W.dims.map(function (d, k) { return { i: k + 1, label: d.label, d: 2, ranked: true, higher: true }; }));
    host.appendChild(tableToggle("Show the full table", dataTable(
      "WBench Full-289 (289 cases, 1,058 turns). 0–100, higher is better; Overall is the unweighted mean of the five dimensions. Bold: best; underlined: second best.",
      cols, [{ rows: W.rows }], function (r) { return !!r[7]; })));
  })();

  /* ---------- WBench per-metric heat table ---------- */
  function hexToRgb(h) { h = h.trim().replace("#", ""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function rampColor(t) {
    var cs = getComputedStyle(document.documentElement);
    var stops = [0, 1, 2, 3, 4].map(function (i) { return hexToRgb(cs.getPropertyValue("--heat-" + i)); });
    t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    var i = Math.min(Math.floor(t), stops.length - 2), f = t - i;
    return stops[i].map(function (c, k) { return Math.round(c + (stops[i + 1][k] - c) * f); });
  }
  function lum(rgb) {
    var a = rgb.map(function (c) { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
  }
  function heat() {
    var host = $("#heat-viz"); if (!host) return;
    var W = D.wbench; host.textContent = "";
    var table = el("table", { class: "heat" });
    table.appendChild(el("caption", { class: "visually-hidden", text: "WBench Full-289 scores on 22 metrics for Murea-Pro and four baselines. Higher is better; the best score in each row is bold and underlined." }));
    table.appendChild(el("thead", null, [el("tr", null, [el("th", { scope: "col", text: "Metric" })].concat(W.heatModels.map(function (m, k) {
      return el("th", { scope: "col", class: k === 0 ? "is-ours" : "", text: m });
    })))]));
    W.heat.forEach(function (grp) {
      var tb = el("tbody");
      tb.appendChild(el("tr", null, [el("th", { scope: "rowgroup", colspan: W.heatModels.length + 1, text: grp[0] })]));
      grp[1].forEach(function (r) {
        var vals = r.slice(1), best = Math.max.apply(null, vals);
        var tr = el("tr", null, [el("th", { scope: "row", text: r[0] })]);
        vals.forEach(function (v, k) {
          var rgb = rampColor((v - 30) / 70);
          var td = el("td", { class: (v === best ? "best " : "") + (k === 0 ? "is-ours" : ""), text: v.toFixed(2) });
          td.style.background = "rgb(" + rgb.join(",") + ")";
          td.style.color = lum(rgb) < 0.3 ? "#ffffff" : "#111111";
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      table.appendChild(tb);
    });
    host.appendChild(table);
  }
  heat();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", heat);

  /* ---------- OpenS2V-Eval and OS2V-Pro ---------- */
  (function s2v() {
    var host = $("#s2v-viz"); if (!host) return;
    var O = D.opens2v, P = D.os2vpro;
    var groupName = { M: "Ours", C: "Commercial", O: "Open-source" };
    var state = { bench: "opens2v", metric: 0, group: "all" };
    var benches = {
      opens2v: {
        title: "OpenS2V-Eval Open-Domain",
        metrics: O.metrics,
        items: O.rows.map(function (r) { return { name: r[0], ours: r[1] === "M", group: r[1], n: r[2], vals: r.slice(3), sub: r[2] < 180 ? r[2] + "/180 clips" : "", meta: (r[1] === "M" ? "" : groupName[r[1]] + ". ") + r[2] + " of 180 clips returned" }; })
      },
      os2vpro: {
        title: "OS2V-Pro",
        metrics: P.metrics,
        items: P.rows.map(function (r) { return { name: r[0], ours: r[1] === "M", group: r[1], vals: r.slice(2), meta: r[1] === "M" ? "" : groupName[r[1]] }; })
      }
    };
    var title = el("p", { class: "viz__title" }), sub = el("p", { class: "viz__sub" });
    var select = el("select", { class: "select", id: "s2v-metric" });
    var controls = el("div", { class: "controls" }, [
      seg([{ key: "opens2v", label: "OpenS2V-Eval" }, { key: "os2vpro", label: "OS2V-Pro" }], "opens2v", function (k) { state.bench = k; state.metric = 0; fillSelect(); render(); }, "Benchmark"),
      el("label", { for: "s2v-metric" }, ["Metric", select]),
      seg([{ key: "all", label: "All" }, { key: "C", label: "Commercial" }, { key: "O", label: "Open-source" }], "all", function (k) { state.group = k; render(); }, "Systems")
    ]);
    select.addEventListener("change", function () { state.metric = +select.value; render(); });
    host.appendChild(controls); host.appendChild(title); host.appendChild(sub); host.appendChild(legend("Murea-Fast"));
    var plotWrap = el("div"); host.appendChild(plotWrap);
    var plots = {};
    Object.keys(benches).forEach(function (k) {
      var w = el("div"); plotWrap.appendChild(w);
      plots[k] = { wrap: w, plot: new DotPlot(w, benches[k].items) };
    });
    function fillSelect() {
      select.textContent = "";
      benches[state.bench].metrics.forEach(function (m, i) { select.appendChild(el("option", { value: i, text: m[1] })); });
      select.value = state.metric;
    }
    function render() {
      var b = benches[state.bench], m = b.metrics[state.metric];
      Object.keys(plots).forEach(function (k) { plots[k].wrap.hidden = k !== state.bench; });
      title.textContent = b.title + ", " + m[1];
      var scale = state.bench === "opens2v" || ["pro", "opens2v", "fid", "intid", "vtf", "bind", "anat", "vq", "iqa", "usable", "atq", "vesa"].indexOf(m[0]) >= 0 ? "0–100" : "native scale";
      sub.textContent = (m[0] === "wall" ? "Seconds per clip under each system's own deployment; API timings include queueing; not ranked. " : scale + ", " + (m[3] ? "higher" : "lower") + " is better. ") +
        (state.bench === "opens2v" ? "Recomputed with a common pipeline." : "Same clips as OpenS2V-Eval; Seedance 2.0 is not included.");
      plots[state.bench].plot.update(function (it) { return it.vals[state.metric]; }, {
        decimals: m[2], higher: m[3], ranked: m[4] !== false, title: b.title + " " + m[1],
        include: function (it) { return state.group === "all" || it.ours || it.group === state.group; }
      });
    }
    fillSelect(); render();

    var oCols = [{ i: 0, label: "Model" }, { i: 2, label: "N" }].concat(O.metrics.map(function (m, k) { return { i: k + 3, label: m[1], d: m[2], ranked: true, higher: m[3] }; }));
    var oGroups = [{ rows: O.rows.filter(function (r) { return r[1] === "M"; }) }, { label: "Commercial systems", rows: O.rows.filter(function (r) { return r[1] === "C"; }) }, { label: "Open-source models", rows: O.rows.filter(function (r) { return r[1] === "O"; }) }];
    host.appendChild(tableToggle("Show the OpenS2V-Eval table", dataTable(
      "OpenS2V-Eval Open-Domain. N is the number of clips returned out of 180 cases. Total = 0.16 Aes + 0.06 Smooth + 0.02 Motion + 0.20 FaceSim + 0.12 Gme + 0.20 Nexus + 0.24 Natural. NaturalScore recomputed for all systems with gpt-4o-2024-11-20. Bold: best; underlined: second best.",
      oCols, oGroups, function (r) { return r[1] === "M"; })));

    var short = { pro: "Pro", opens2v: "OpenS2V", fid: "Fid", intid: "IntID", decay: "Decay", vtf: "VTF", bind: "Bind", anat: "Anat", limb: "Limb", vq: "VQ", iqa: "IQA", sharp: "Sharp", flick: "Flick", usable: "Usable", atq: "ATQ", vesa: "VESA", wall: "Wall (s)" };
    var pCols = [{ i: 0, label: "Model" }].concat(P.metrics.map(function (m, k) { return { i: k + 2, label: short[m[0]], title: m[1], d: m[2], ranked: m[4], higher: m[3] }; }));
    var pGroups = [{ rows: P.rows.filter(function (r) { return r[1] === "M"; }) }, { label: "Commercial systems", rows: P.rows.filter(function (r) { return r[1] === "C"; }) }, { label: "Open-source models", rows: P.rows.filter(function (r) { return r[1] === "O"; }) }];
    host.appendChild(tableToggle("Show the OS2V-Pro table", dataTable(
      "OS2V-Pro on the OpenS2V-Eval Open-Domain clips. Pro is the aggregate: identity, instruction following, rendering, audio-track completeness, and set-level measures, 20% each. Decay, Limb, Sharp, and Flick keep their native scales; ↓ marks lower is better. ATQ and VESA are excluded from the aggregate; latency is not ranked. Bold: best; underlined: second best.",
      pCols, pGroups, function (r) { return r[1] === "M"; })));
  })();

  /* ---------- inference latency ---------- */
  (function latency() {
    var host = $("#latency-viz"); if (!host) return;
    var L = D.latency, base = L.baseline.value;
    function group(h, note, axisMax, tickStep, rows) {
      var g = el("div", { class: "bars__group" }, [el("h4", { text: h }), el("p", { text: note })]);
      var wrap = el("div", { class: "bars", role: "img", "aria-label": h + ": " + rows.map(function (r) { return r.label + " " + r.value + " s"; }).join("; ") });
      rows.forEach(function (r) {
        var pct = Math.min(r.value / axisMax, 1) * 100;
        var track = el("div", { class: "bars__track" });
        if (r.ghost != null) track.appendChild(el("span", { class: "bars__ghost", style: "width:" + Math.min(r.ghost / axisMax, 1) * 100 + "%" }));
        track.appendChild(el("span", { class: "bars__fill", style: "width:calc(" + pct + "% - 2px)" }));
        var inside = pct > 78;
        var val = el("span", { class: "bars__val", text: (r.value < 10 ? r.value.toFixed(2) : r.value.toFixed(1)) + " s" });
        if (inside) { val.style.right = (100 - pct) + "%"; val.style.paddingRight = ".5rem"; val.style.color = "#fff"; }
        else val.style.left = pct + "%";
        track.appendChild(val);
        var row = el("div", { class: "bars__row" + (r.ours ? " is-ours" : ""), "aria-hidden": "true" }, [
          el("div", { class: "bars__lab" }, [el("b", { text: r.label }), el("span", { text: r.note })]),
          track,
          el("div", { class: "bars__sp" }, [el("b", { text: r.total }), el("span", { text: r.stepNote || "" })])
        ]);
        row.addEventListener("pointerenter", function (e) {
          tipShow(e, (r.value < 10 ? r.value.toFixed(2) : r.value.toFixed(1)) + " s", [r.label, r.value === base ? "50-step baseline" : (base - r.value).toFixed(r.value < 10 ? 2 : 1) + " s saved vs. baseline", r.value === base ? "" : r.total + " faster than baseline"]);
        });
        row.addEventListener("pointermove", tipMove);
        row.addEventListener("pointerleave", tipHide);
        wrap.appendChild(row);
      });
      var ticks = el("div", { class: "ticks" });
      for (var t = 0; t <= axisMax; t += tickStep) ticks.appendChild(el("span", { style: "left:" + (t / axisMax) * 100 + "%", text: String(t) }));
      wrap.appendChild(el("div", { class: "bars__axis", "aria-hidden": "true" }, [el("span"), ticks, el("span")]));
      g.appendChild(wrap);
      return g;
    }
    var last = L.steps[L.steps.length - 1];
    host.appendChild(el("p", { class: "viz__title", text: "End-to-end latency for one 5.04 s clip" }));
    host.appendChild(el("p", { class: "viz__sub", text: "Eight NVIDIA H20 GPUs, fused kernels enabled throughout. Lower is better." }));
    host.appendChild(group("Baseline and fully optimized", "Seconds, 0–250 scale.", 250, 50, [
      { label: L.baseline.label, note: L.baseline.note, value: base, total: "1.0×" },
      { label: "Fully optimized", note: "All four optimizations", value: last.value, total: (base / last.value).toFixed(1) + "×", stepNote: "vs. baseline", ours: true }
    ]));
    var prevV = base;
    host.appendChild(group("Each optimization, applied in sequence", "Seconds, 0–25 scale. Each row keeps every optimization above it; the pale segment is the time the new step removed.", 25, 5, L.steps.map(function (s, i) {
      var r = { label: s.label, note: s.note, value: s.value, ghost: i === 0 ? null : prevV, total: (base / s.value).toFixed(1) + "×", stepNote: "step " + s.step, ours: i === L.steps.length - 1 };
      if (i === 0) r.note = s.note + ", from " + base + " s";
      prevV = s.value; return r;
    })));
    var rows = [["Baseline (50 NFE, FlashAttention-3)", base, null, "1.0×", "—"]];
    L.steps.forEach(function (s) { rows.push([s.label + " (" + s.note + ")", s.value, +(base - s.value).toFixed(2), (base / s.value).toFixed(1) + "×", s.step]); });
    host.appendChild(tableToggle("Show the table", dataTable(
      "Cumulative inference acceleration on eight NVIDIA H20 GPUs; each row keeps all preceding optimizations.",
      [{ i: 0, label: "Configuration" }, { i: 1, label: "Latency (s)", d: 2 }, { i: 2, label: "Saved vs. baseline (s)", d: 2 }, { i: 3, label: "Total speedup" }, { i: 4, label: "Step speedup" }],
      [{ rows: rows }], function (r) { return r[1] === last.value; })));
  })();

  /* ---------- BibTeX copy ---------- */
  var copyBtn = $("#copy-bib");
  if (copyBtn) copyBtn.addEventListener("click", function () {
    var text = $("#bibtex").textContent;
    function done() { copyBtn.textContent = "Copied"; setTimeout(function () { copyBtn.textContent = "Copy BibTeX"; }, 1800); }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done);
    else {
      var r = document.createRange(); r.selectNodeContents($("#bibtex"));
      var s = getSelection(); s.removeAllRanges(); s.addRange(r);
      try { document.execCommand("copy"); done(); } catch (e) { /* selection stays for manual copy */ }
    }
  });
})();
