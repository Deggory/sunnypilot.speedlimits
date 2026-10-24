"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

  // web/src/common/dom.ts
  function byId(id) {
    const element = document.getElementById(id);
    if (element === null) throw new Error("Elemento mancante nella pagina: #" + id);
    return element;
  }
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== void 0) element.textContent = text;
    return element;
  }

  // web/src/common/theme.ts
  var KEY = "nnslr-theme";
  var CHANGED = "nnslr-theme";
  var root = document.documentElement;
  function isDark() {
    const theme = root.dataset["theme"];
    if (theme === "light") return false;
    if (theme === "auto") return !matchMedia("(prefers-color-scheme: light)").matches;
    return true;
  }
  function setTheme(theme) {
    try {
      localStorage.setItem(KEY, theme);
    } catch {
    }
    if (theme === "dark") delete root.dataset["theme"];
    else root.dataset["theme"] = theme;
    document.dispatchEvent(new Event(CHANGED));
  }
  function onThemeChange(listener) {
    document.addEventListener(CHANGED, listener);
  }
  function initThemeSwitch() {
    const button = document.getElementById("theme-toggle");
    if (button === null) return;
    const show = () => {
      const dark = isDark();
      const title = dark ? "Passa al tema chiaro" : "Passa al tema scuro";
      button.classList.toggle("is-light", !dark);
      button.setAttribute("aria-pressed", String(!dark));
      button.title = title;
      button.setAttribute("aria-label", title);
    };
    button.addEventListener("click", () => setTheme(isDark() ? "light" : "dark"));
    onThemeChange(show);
    show();
  }

  // web/src/common/api.ts
  function messageOf(error) {
    return error instanceof Error ? error.message : String(error);
  }

  // web/src/review/dom.ts
  var SVG_NS = "http://www.w3.org/2000/svg";
  function svgById(id) {
    const element = document.getElementById(id);
    if (!(element instanceof SVGElement)) throw new Error("Elemento SVG mancante nella pagina: #" + id);
    return element;
  }
  function one(selector) {
    const element = document.querySelector(selector);
    if (element === null) throw new Error("Elemento mancante nella pagina: " + selector);
    return element;
  }
  function svgNode(tag, attributes) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
    return element;
  }
  function datasetOf(target) {
    if (target instanceof HTMLElement || target instanceof SVGElement) return target.dataset;
    return null;
  }
  function narrowLayout() {
    return window.matchMedia("(max-width: 980px)").matches;
  }
  function coarsePointer() {
    return window.matchMedia("(pointer: coarse)").matches;
  }
  function readLocal(key, fallback) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  }
  function writeLocal(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
    }
  }
  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  // web/src/review/catalog.ts
  var FAMILIES = [
    { id: "maximum_speed", label: "Limite massimo", hasValue: true, minor: false },
    { id: "cancellation", label: "Fine limitazione", hasValue: true, minor: false },
    { id: "unreadable", label: "Illeggibile", hasValue: false, minor: false },
    { id: "variable_display", label: "Segnale variabile", hasValue: true, minor: false },
    { id: "zone", label: "Zona", hasValue: true, minor: false },
    { id: "road_marking_candidate", label: "Numero a terra", hasValue: true, minor: true },
    { id: "other_sign", label: "Altro cartello", hasValue: false, minor: true },
    { id: "not_a_sign", label: "Non è un cartello", hasValue: false, minor: true }
  ];
  var QUICK_VALUES = [20, 30, 40, 50, 60, 70, 80, 100, 120];
  var ROADS = [
    { id: "own_road", label: "La nostra" },
    { id: "other_road", label: "Un’altra" },
    { id: "unresolved", label: "Da chiarire" }
  ];
  var SELECT_FIELDS = ["visibility", "occlusion", "supplementary_panel"];
  var BOOLEAN_FIELDS = ["temporary", "variable_display"];
  var UNSURE = { id: "unsure", label: "Incerto", short: "?" };
  var HINT_TYPES = [
    { id: "maximum_speed", label: "Limite massimo", short: "limite" },
    { id: "cancellation", label: "Fine limitazione", short: "fine" },
    { id: "zone", label: "Zona", short: "zona" },
    { id: "zone_end", label: "Fine zona", short: "fine zona" },
    { id: "variable_display", label: "Segnale variabile", short: "display" },
    { id: "road_marking", label: "Numero a terra", short: "a terra" },
    UNSURE
  ];
  var CONFIDENCES = /* @__PURE__ */ new Map([["high", "sicura"], ["medium", "media"], ["low", "bassa"]]);
  function isSpeedFamily(family) {
    return family === "maximum_speed" || family === "zone" || family === "variable_display";
  }
  function familyOf(id) {
    return FAMILIES.find((family) => family.id === id) ?? null;
  }
  function familyHasValue(id) {
    return familyOf(id)?.hasValue ?? false;
  }
  function hasValue(d) {
    return d.value_kph !== null && d.value_kph !== void 0;
  }
  function describe(d) {
    const family = familyOf(d.family);
    let text = family ? family.label : String(d.family || "Senza tipo");
    if (hasValue(d)) text += " · " + d.value_kph + " km/h";
    return text;
  }
  function shortName(d) {
    const v = d.value_kph;
    const known = hasValue(d);
    switch (d.family) {
      case "maximum_speed":
        return known ? String(v) : "limite ?";
      case "zone":
        return known ? "zona " + v : "zona";
      case "variable_display":
        return known ? "variabile " + v : "variabile";
      case "road_marking_candidate":
        return known ? "a terra " + v : "a terra";
      case "cancellation":
        return known ? "fine " + v : "fine";
      case "other_sign":
        return "altro";
      case "not_a_sign":
        return "no";
      default:
        return "?";
    }
  }
  function boxColor(d) {
    if (isSpeedFamily(d.family)) return "var(--ov-speed)";
    if (d.family === "cancellation") return "var(--ov-cancel)";
    if (d.family === "not_a_sign") return "var(--ov-none)";
    return "var(--ov-other)";
  }
  function hintTypeOf(id) {
    return HINT_TYPES.find((type) => type.id === id) ?? UNSURE;
  }
  function confidenceLabel(id) {
    return CONFIDENCES.get(id) ?? String(id);
  }
  function sideLabel(side) {
    if (side === "left") return "a sinistra";
    if (side === "right") return "a destra";
    if (side === "center" || side === "centre") return "al centro";
    if (side === "overhead") return "sopra la strada";
    return side;
  }
  function hintLabel(h) {
    const type = hintTypeOf(h.type);
    let what = type.short;
    if (hasValue(h)) {
      what = h.type === "maximum_speed" || h.type === "unsure" ? String(h.value_kph) : type.short + " " + h.value_kph;
    } else if (h.type === "unsure") {
      what = "incerto";
    }
    return what + "?";
  }
  function familyForHint(type) {
    switch (type) {
      case "maximum_speed":
        return "maximum_speed";
      case "cancellation":
        return "cancellation";
      case "zone":
        return "zone";
      case "zone_end":
        return "cancellation";
      case "variable_display":
        return "variable_display";
      case "road_marking":
        return "road_marking_candidate";
      default:
        return "unreadable";
    }
  }

  // web/src/review/state.ts
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  var raw = JSON.parse(byId("payload").textContent || "{}");
  var runId = String(raw.run_id);
  var routeId = raw.route_id ?? "";
  var rows = Array.isArray(raw.frames) ? raw.frames : [];
  var routes = Array.isArray(raw.routes) ? raw.routes : [];
  var hintsByFrame = isRecord(raw.hints) ? raw.hints : {};
  var hintStats = isRecord(raw.hint_stats) ? raw.hint_stats : null;
  var routeHasHints = Object.keys(hintsByFrame).length > 0;
  var apiBase = raw.server ? "/api/route/" + encodeURIComponent(routeId) : null;
  var storageKey = "nnslr-review-v1-" + runId;
  var lastKeyStorage = "nnslr-review-last-" + runId;
  var state = {
    edits: {},
    suspicious: isRecord(raw.suspicious) ? raw.suspicious : {},
    currentKey: null,
    selected: -1,
    drawing: false,
    drag: null,
    ignoreClickUntil: 0,
    filter: "todo",
    filterNote: "",
    showHints: true,
    inFlight: /* @__PURE__ */ new Set(),
    notice: null,
    storageWarning: "",
    serverWarning: "",
    retrying: false,
    withdrawArmed: null,
    saveCount: 0,
    lastSave: null,
    server: "offline"
  };
  var rowIndex = new Map(rows.map((row, index) => [row.frame_key, index]));
  function indexOf(key) {
    return rowIndex.get(key) ?? -1;
  }
  function rowOf(key) {
    const index = rowIndex.get(key);
    return index === void 0 ? null : rows[index] ?? null;
  }
  function hintsOf(key) {
    const list = hintsByFrame[key];
    return Array.isArray(list) ? list : [];
  }
  var renderer = () => void 0;
  function setRenderer(draw) {
    renderer = draw;
  }
  function render() {
    renderer();
  }
  function store() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state.edits));
      state.storageWarning = "";
    } catch {
      state.storageWarning = "Salvataggio nel browser non disponibile: scarica il JSONL prima di chiudere.";
    }
  }

  // web/src/review/frames.ts
  function currentRow() {
    return state.currentKey === null ? null : rowOf(state.currentKey);
  }
  function proposalsOf(row) {
    return Array.isArray(row.detections) ? row.detections : [];
  }
  function workOf(row) {
    const key = row.frame_key;
    let work = state.edits[key];
    if (!work) {
      const detections = clone(proposalsOf(row));
      detections.sort((a, b) => a.bbox_xyxy[0] - b.bbox_xyxy[0]);
      work = { review_state: "pending", detections, saved: true, revision: 0 };
      state.edits[key] = work;
    }
    return work;
  }
  function selectedDetection() {
    const row = currentRow();
    if (!row) return null;
    return workOf(row).detections[state.selected] ?? null;
  }
  function fromServer(item) {
    return {
      review_state: item.confirmed ? "reviewed" : "pending",
      detections: clone(item.detections ?? []),
      reviewer: item.reviewer,
      reviewer_kind: item.reviewer_kind || "human",
      channel: item.channel,
      review_timestamp: item.review_timestamp,
      site_group: item.site_group,
      encounter_id: item.encounter_id,
      revision: item.revision,
      saved: true,
      error: null,
      server_confirmed: item.confirmed,
      base: clone(item)
    };
  }
  function sameAsServer(s) {
    if (!s || !s.base) return false;
    if (JSON.stringify(s.detections) !== JSON.stringify(s.base.detections || [])) return false;
    if ((s.site_group ?? null) !== (s.base.site_group ?? null)) return false;
    return (s.encounter_id ?? null) === (s.base.encounter_id ?? null);
  }
  function isConfirmed(key) {
    const s = state.edits[key];
    if (!s) return false;
    if (apiBase) return s.server_confirmed === true;
    return s.review_state === "reviewed";
  }
  function reasonsOf(key) {
    if (isConfirmed(key)) return [];
    const list = state.suspicious[key];
    return Array.isArray(list) ? list : [];
  }
  function frameStatus(key) {
    if (apiBase && state.inFlight.has(key)) return "saving";
    const s = state.edits[key];
    if (!s) return "todo";
    if (s.review_state === "reviewed") {
      if (!apiBase) return "done";
      if (s.saved === true && s.server_confirmed === true) return "done";
      return "error";
    }
    if (s.saved === false) return "dirty";
    return "todo";
  }
  function aiConfirmed(key) {
    const s = state.edits[key];
    return Boolean(apiBase && s && s.reviewer_kind === "ai" && frameStatus(key) === "done" && sameAsServer(s));
  }
  function statusLabel(key) {
    const status = frameStatus(key);
    const s = state.edits[key];
    if (status === "done" && aiConfirmed(key)) return "Confermato · AI";
    if (status === "done") return "Confermato";
    if (status === "saving") return "Salvataggio…";
    if (status === "error") return s && s.error ? "Non salvato" : "In attesa";
    if (status === "dirty") return s && s.server_confirmed ? "Da riconfermare" : "Da confermare";
    if (reasonsOf(key).length > 0) return "Sospetto";
    return "Da rivedere";
  }
  function doubtsOf(key) {
    const out = { unknown: 0, unreadable: 0, road: 0 };
    const s = state.edits[key];
    if (!s || !isConfirmed(key)) return out;
    for (const d of s.detections) {
      const speed = isSpeedFamily(d.family);
      if (d.family === "unreadable") {
        out.unreadable += 1;
      } else if ((speed || d.family === "road_marking_candidate") && !hasValue(d)) {
        out.unknown += 1;
      }
      if (speed && (!d.applicability || d.applicability === "unresolved")) out.road += 1;
    }
    return out;
  }
  function isQualityFilter(filter) {
    return filter === "unknown" || filter === "unreadable" || filter === "road";
  }
  function confirmedFilter(filter) {
    return isQualityFilter(filter) || filter === "ai";
  }
  function isFilter(value) {
    return value === "todo" || value === "suspicious" || value === "proposals" || value === "all" || value !== null && value !== void 0 && confirmedFilter(value);
  }
  function matchesFilter(row, filter) {
    const key = row.frame_key;
    switch (filter) {
      case "todo":
        return frameStatus(key) !== "done";
      case "suspicious":
        return reasonsOf(key).length > 0;
      case "proposals":
        return proposalsOf(row).length > 0;
      case "ai":
        return aiConfirmed(key);
      case "unknown":
        return doubtsOf(key).unknown > 0;
      case "unreadable":
        return doubtsOf(key).unreadable > 0;
      case "road":
        return doubtsOf(key).road > 0;
      default:
        return true;
    }
  }
  function visibleRows() {
    return rows.filter((row) => row.frame_key === state.currentKey || matchesFilter(row, state.filter));
  }
  function unsavedKeys() {
    return rows.filter((row) => {
      const s = state.edits[row.frame_key];
      return s && s.saved === false && s.review_state === "reviewed";
    }).map((row) => row.frame_key);
  }

  // web/src/review/timeline.ts
  function ticks() {
    return byId("timeline").querySelectorAll("button");
  }
  function buildTimeline() {
    const box = byId("timeline");
    box.replaceChildren();
    let lastSegment = null;
    let strip = null;
    let longest = 1;
    for (const row of rows) {
      if (strip === null || row.segment_index !== lastSegment) {
        const segment = node("div", "tl-seg");
        strip = node("div", "tl-ticks");
        segment.append(node("span", "tl-label", "seg " + row.segment_index), strip);
        box.append(segment);
        lastSegment = row.segment_index;
      }
      const tick = node("button");
      tick.type = "button";
      tick.tabIndex = -1;
      tick.dataset["key"] = row.frame_key;
      strip.append(tick);
      longest = Math.max(longest, strip.childElementCount);
    }
    box.style.setProperty("--tl-cols", String(longest));
  }
  function renderTimeline() {
    for (const tick of ticks()) {
      const key = tick.dataset["key"] ?? "";
      const row = rowOf(key);
      if (row === null) continue;
      const status = frameStatus(key);
      let cls = "";
      if (status === "done") {
        const doubts = doubtsOf(key);
        cls = doubts.unknown + doubts.unreadable > 0 ? "done unsure" : "done";
        if (aiConfirmed(key)) cls += " ai";
      } else if (status === "error") {
        cls = "error";
      } else if (status === "dirty" || status === "saving") {
        cls = "dirty";
      } else if (reasonsOf(key).length > 0) {
        cls = "sus";
      } else if (proposalsOf(row).length > 0 || hintsOf(key).length > 0) {
        cls = "has";
      }
      if (key === state.currentKey) cls += " cur";
      if (tick.className !== cls) tick.className = cls;
      const label2 = "Segmento " + row.segment_index + " · immagine " + row.output_index + " · " + statusLabel(key).toLowerCase();
      if (tick.title !== label2) {
        tick.title = label2;
        tick.setAttribute("aria-label", label2);
      }
    }
  }
  function scrollTimelineToCurrent() {
    for (const tick of ticks()) {
      if (tick.dataset["key"] !== state.currentKey) continue;
      const strip = tick.parentElement;
      if (strip === null || strip.scrollWidth <= strip.clientWidth) return;
      const left = tick.offsetLeft - strip.offsetLeft;
      if (left >= strip.scrollLeft + 40 && left <= strip.scrollLeft + strip.clientWidth - 40) return;
      const target = left - strip.clientWidth / 2;
      strip.scrollLeft = target < 0 ? 0 : target;
    }
  }
  function renderQuality() {
    let unknown = 0;
    let unreadable = 0;
    let road = 0;
    for (const row of rows) {
      const doubts = doubtsOf(row.frame_key);
      if (doubts.unknown > 0) unknown += 1;
      if (doubts.unreadable > 0) unreadable += 1;
      if (doubts.road > 0) road += 1;
    }
    setQualityChip(byId("qUnknown"), unknown, "senza numero", true);
    setQualityChip(byId("qUnreadable"), unreadable, "con illeggibili", true);
    setQualityChip(byId("qRoad"), road, "strada non indicata", false);
  }
  function setQualityChip(chip, count, label2, warn) {
    chip.textContent = count + " " + label2;
    chip.dataset["count"] = String(count);
    if (count > 0 && warn) chip.className = "qchip some";
    else if (count > 0) chip.className = "qchip";
    else chip.className = "qchip zero";
    chip.setAttribute("aria-pressed", state.filter === chip.dataset["filter"] ? "true" : "false");
  }
  function initTimeline(open) {
    byId("tlToggle").addEventListener("click", () => {
      const opened = !byId("workspace").classList.contains("tl-open");
      setTimelineOpen(opened);
      writeLocal("nnslr-timeline-open", opened ? "1" : "0");
      if (opened) scrollTimelineToCurrent();
    });
    byId("timeline").addEventListener("click", (event) => {
      const target = event.target;
      const key = target instanceof HTMLElement ? target.dataset["key"] : void 0;
      if (key) open(key);
    });
  }
  function setTimelineOpen(opened) {
    byId("workspace").classList.toggle("tl-open", opened);
    byId("tlToggle").setAttribute("aria-expanded", opened ? "true" : "false");
  }

  // web/src/review/navigation.ts
  function go(step) {
    const list = visibleRows();
    const index = list.findIndex((row) => row.frame_key === state.currentKey);
    const target = list[index + step];
    if (index + step < 0 || target === void 0) return;
    openFrame(target.frame_key);
  }
  function openFrame(key) {
    state.currentKey = key;
    state.selected = -1;
    state.drawing = false;
    state.drag = null;
    state.filterNote = "";
    state.withdrawArmed = null;
    writeLocal(lastKeyStorage, key);
    render();
    scrollTimelineToCurrent();
    revealStage();
  }
  function revealStage() {
    if (!narrowLayout()) return;
    const header = one(".topbar");
    const top = getComputedStyle(header).position === "sticky" ? header.getBoundingClientRect().bottom : 0;
    const rect = byId("stage").getBoundingClientRect();
    if (rect.top < top || rect.top > window.innerHeight * 0.6) {
      window.scrollTo(0, window.scrollY + rect.top - top - 8);
    }
  }
  function nextAfterConfirm(key) {
    for (const row of rows.slice(indexOf(key) + 1)) {
      if (state.filter === "all") return row.frame_key;
      if (confirmedFilter(state.filter)) {
        if (matchesFilter(row, state.filter)) return row.frame_key;
        continue;
      }
      if (frameStatus(row.frame_key) === "done") continue;
      if (matchesFilter(row, state.filter)) return row.frame_key;
    }
    return null;
  }
  function advanceAfterConfirm(key, savedText) {
    const next = nextAfterConfirm(key);
    if (next === null) {
      state.notice = { key, text: savedText + " Non restano fotogrammi da rivedere con questo filtro.", error: false };
      render();
      return;
    }
    state.notice = { key: next, text: savedText + " Ecco il prossimo.", error: false };
    openFrame(next);
  }
  function setFilter(filter) {
    state.filter = filter;
    writeLocal("nnslr-review-filter", filter);
    const row = currentRow();
    if (row === null || matchesFilter(row, filter)) {
      state.filterNote = "";
      render();
      return;
    }
    const start2 = indexOf(row.frame_key);
    const target = rows.slice(start2 + 1).find((item) => matchesFilter(item, filter)) ?? rows.slice(0, Math.max(start2, 0)).find((item) => matchesFilter(item, filter));
    if (target === void 0) {
      if (filter === "suspicious") state.filterNote = "Nessun fotogramma sospetto da rivedere.";
      else if (isQualityFilter(filter)) state.filterNote = "Nessun fotogramma confermato con questo dubbio.";
      else if (filter === "ai") state.filterNote = "Nessun fotogramma confermato da un’AI.";
      else state.filterNote = "Nessun fotogramma corrisponde a questo filtro.";
      render();
      return;
    }
    openFrame(target.frame_key);
  }
  function pickStartFrame() {
    const last = readLocal(lastKeyStorage, null);
    if (last !== null && rowOf(last)) return last;
    const first = rows.find((row) => matchesFilter(row, state.filter)) ?? rows[0];
    return first ? first.frame_key : null;
  }

  // web/src/review/server.ts
  var SaveError = class extends Error {
    constructor(message, retry) {
      super(message);
      __publicField(this, "retry", retry);
    }
  };
  async function call(method, url, body) {
    const headers = { "Accept": "application/json" };
    const options = { method, headers, credentials: "same-origin", cache: "no-store" };
    if (body !== void 0) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new SaveError("server non raggiungibile.", true);
    }
    let result = null;
    try {
      result = await response.json();
    } catch {
      result = null;
    }
    if (!response.ok) {
      const problem = result;
      throw new SaveError(
        problem && problem.message ? problem.message : "errore HTTP " + response.status + ".",
        response.status >= 500
      );
    }
    return result;
  }
  function takeSuspicious(result) {
    const flags = result?.suspicious;
    if (flags && typeof flags === "object" && !Array.isArray(flags)) state.suspicious = flags;
  }
  function reviewerName() {
    return byId("reviewer").value.trim();
  }
  async function send(key) {
    const s = state.edits[key];
    if (!apiBase || !s || state.inFlight.has(key)) return false;
    const version = s.version || 0;
    state.inFlight.add(key);
    state.lastSave = { key, result: "saving", revision: s.revision || 0 };
    render();
    let ok = false;
    try {
      const result = await call("POST", apiBase + "/frame/" + key, {
        run_id: runId,
        reviewer: s.reviewer || reviewerName(),
        confirmed: true,
        detections: s.detections,
        site_group: s.site_group === void 0 ? null : s.site_group,
        encounter_id: s.encounter_id === void 0 ? null : s.encounter_id,
        base_revision: s.revision || 0
      });
      if (state.edits[key] === s && (s.version || 0) === version) {
        state.edits[key] = fromServer(result.review);
      } else {
        s.revision = result.review.revision;
        s.server_confirmed = result.review.confirmed;
        s.base = clone(result.review);
      }
      takeSuspicious(result);
      ok = true;
    } catch (error) {
      s.error = messageOf(error);
      s.retry = error instanceof SaveError && error.retry;
    } finally {
      state.inFlight.delete(key);
      state.saveCount += 1;
      const saved = ok ? state.edits[key] : s;
      state.lastSave = { key, result: ok ? "ok" : "error", revision: saved?.revision || 0 };
      store();
      render();
    }
    return ok;
  }
  async function retryUnsaved(all) {
    if (!apiBase || state.retrying) return;
    state.retrying = true;
    try {
      for (const key of unsavedKeys()) {
        if (all || state.edits[key]?.retry !== false) await send(key);
      }
    } finally {
      state.retrying = false;
    }
  }
  async function loadServer() {
    if (!apiBase) return;
    try {
      const body = await call("GET", apiBase + "/reviews");
      if (body.run_id !== runId) {
        state.serverWarning = "Il report di questa route è stato rigenerato: ricarica la pagina.";
        state.server = "stale";
        render();
        return;
      }
      const onServer = /* @__PURE__ */ new Set();
      for (const item of body.reviews) {
        onServer.add(item.frame_key);
        const local = state.edits[item.frame_key];
        if (local && local.saved === false && (local.revision || 0) === item.revision) {
          local.server_confirmed = item.confirmed;
          local.base = clone(item);
        } else {
          state.edits[item.frame_key] = fromServer(item);
        }
      }
      for (const row of rows) {
        const key = row.frame_key;
        const s = state.edits[key];
        if (!s || onServer.has(key)) continue;
        s.server_confirmed = false;
        s.base = null;
        if (s.review_state === "reviewed" && s.saved !== false) {
          s.saved = false;
          s.revision = 0;
        } else if (s.saved === void 0) {
          s.saved = true;
        }
      }
      takeSuspicious(body);
      state.serverWarning = "";
      state.server = "ready";
      store();
      render();
      void retryUnsaved(true);
    } catch (error) {
      state.serverWarning = "Revisioni non lette dal server: " + messageOf(error) + " Le modifiche restano in questo browser.";
      state.server = "error";
      render();
    }
  }

  // web/src/review/confirm.ts
  function needReviewer(key) {
    const input = byId("reviewer");
    input.classList.add("need");
    input.focus();
    state.notice = { key, text: "Scrivi nome o sigla del revisore, poi conferma.", error: true };
    state.lastSave = { key, result: "need-reviewer", revision: 0 };
    render();
  }
  function timeNow() {
    return (/* @__PURE__ */ new Date()).toLocaleTimeString("it-CH", { hour: "2-digit", minute: "2-digit" });
  }
  async function confirmFrame() {
    const row = currentRow();
    if (!row) return;
    const key = row.frame_key;
    if (state.inFlight.has(key)) return;
    const current = state.edits[key];
    if (apiBase && frameStatus(key) === "done" && current && sameAsServer(current)) {
      state.lastSave = { key, result: "unchanged", revision: current.revision || 0 };
      go(1);
      if (state.currentKey === key) {
        state.notice = { key, text: "Già confermato: non ci sono altri fotogrammi con questo filtro.", error: false };
      }
      render();
      return;
    }
    const reviewer = reviewerName();
    if (!reviewer) {
      needReviewer(key);
      return;
    }
    writeLocal("nnslr-reviewer", reviewer);
    const s = workOf(row);
    s.review_state = "reviewed";
    s.reviewer = reviewer;
    s.reviewer_kind = "human";
    s.review_timestamp = (/* @__PURE__ */ new Date()).toISOString();
    s.saved = false;
    s.error = null;
    s.retry = true;
    s.version = (s.version || 0) + 1;
    state.drawing = false;
    if (!apiBase) {
      s.saved = true;
      s.base = {
        detections: clone(s.detections),
        site_group: s.site_group === void 0 ? null : s.site_group,
        encounter_id: s.encounter_id === void 0 ? null : s.encounter_id,
        reviewer: s.reviewer,
        review_timestamp: s.review_timestamp
      };
      store();
      advanceAfterConfirm(key, "Confermato " + key + " in questo browser.");
      return;
    }
    state.notice = null;
    store();
    render();
    const ok = await send(key);
    if (ok && state.currentKey === key) advanceAfterConfirm(key, "Salvato " + key + " alle " + timeNow() + ".");
  }
  async function withdrawFrame() {
    const row = currentRow();
    if (!row || !apiBase) return;
    const key = row.frame_key;
    if (state.inFlight.has(key)) return;
    const reviewer = reviewerName();
    if (!reviewer) {
      needReviewer(key);
      return;
    }
    if (state.withdrawArmed !== key) {
      state.withdrawArmed = key;
      const text = "Premi “Sì, annulla” per togliere la conferma: il fotogramma non verrà importato.";
      state.notice = { key, text, error: false, warn: true };
      render();
      return;
    }
    state.withdrawArmed = null;
    const s = workOf(row);
    state.inFlight.add(key);
    render();
    try {
      const result = await call(
        "POST",
        apiBase + "/frame/" + key,
        { run_id: runId, reviewer, confirmed: false, base_revision: s.revision || 0 }
      );
      const saved = fromServer(result.review);
      state.edits[key] = saved;
      takeSuspicious(result);
      state.notice = { key, text: "Conferma annullata: il fotogramma non verrà importato.", error: false };
      state.lastSave = { key, result: "withdrawn", revision: saved.revision || 0 };
    } catch (error) {
      state.notice = { key, text: "Annullamento non riuscito: " + messageOf(error), error: true };
      state.lastSave = { key, result: "error", revision: s.revision || 0 };
    } finally {
      state.inFlight.delete(key);
      state.saveCount += 1;
      store();
      render();
    }
  }
  function discardChanges() {
    const row = currentRow();
    if (!row) return;
    const key = row.frame_key;
    const s = state.edits[key];
    if (!s || state.inFlight.has(key)) return;
    if (s.base) {
      if (apiBase) {
        state.edits[key] = fromServer(s.base);
      } else {
        state.edits[key] = {
          review_state: "reviewed",
          detections: clone(s.base.detections || []),
          site_group: s.base.site_group === void 0 ? null : s.base.site_group,
          encounter_id: s.base.encounter_id === void 0 ? null : s.base.encounter_id,
          reviewer: s.base.reviewer,
          review_timestamp: s.base.review_timestamp,
          saved: true,
          revision: 0,
          base: s.base
        };
      }
    } else {
      delete state.edits[key];
    }
    state.selected = -1;
    state.notice = null;
    state.withdrawArmed = null;
    byId("valueError").hidden = true;
    store();
    render();
  }

  // web/src/review/geometry.ts
  function isBox(value) {
    return Array.isArray(value) && value.length === 4 && value.every((item) => typeof item === "number");
  }
  function sameBox(a, b) {
    if (!isBox(a) || !isBox(b)) return false;
    return Math.abs(a[0] - b[0]) <= 0.01 && Math.abs(a[1] - b[1]) <= 0.01 && Math.abs(a[2] - b[2]) <= 0.01 && Math.abs(a[3] - b[3]) <= 0.01;
  }
  function boxesOverlap(a, b) {
    if (a[2] < b[0] || b[2] < a[0]) return false;
    return !(a[3] < b[1] || b[3] < a[1]);
  }
  function overlap(a, b) {
    const x1 = Math.max(a[0], b[0]);
    const y1 = Math.max(a[1], b[1]);
    const x2 = Math.min(a[2], b[2]);
    const y2 = Math.min(a[3], b[3]);
    if (x2 <= x1 || y2 <= y1) return 0;
    const shared = (x2 - x1) * (y2 - y1);
    const smaller = Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
    if (smaller <= 0) return 0;
    return shared / smaller;
  }
  function boxArea(b) {
    return Math.abs((b[2] - b[0]) * (b[3] - b[1]));
  }
  function rectOf(b) {
    return { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1] };
  }
  function labelWidth(text, fontSize) {
    return text.length * fontSize * 0.6;
  }
  function labelClash(placed, x, y, width, height) {
    return placed.some((p) => x < p.x + p.w && p.x < x + width && y - height < p.y && p.y - p.h < y);
  }
  function freeLabelY(placed, x, y, width, height, step) {
    let candidate = y;
    for (let attempt = 0; attempt < 4; attempt++) {
      if (!labelClash(placed, x, candidate, width, height)) return candidate;
      candidate += step;
    }
    return candidate;
  }

  // web/src/review/edits.ts
  function changed(row) {
    const s = workOf(row);
    s.review_state = "pending";
    s.saved = false;
    s.error = null;
    s.version = (s.version || 0) + 1;
    state.withdrawArmed = null;
    if (state.notice && state.notice.key === row.frame_key) state.notice = null;
    store();
    render();
  }
  function editSelected(change) {
    const d = selectedDetection();
    const row = currentRow();
    if (!d || !row) return;
    change(d);
    changed(row);
  }
  function setFamily(id) {
    editSelected((d) => {
      d.family = id;
      if (!familyHasValue(id)) d.value_kph = null;
      d.value_state = d.value_kph === null || d.value_kph === void 0 ? "unknown" : "reviewed";
      byId("valueError").hidden = true;
    });
  }
  function setValue(value) {
    editSelected((d) => {
      if (!familyHasValue(d.family)) d.family = "maximum_speed";
      d.value_kph = value;
      d.value_state = value === null ? "unknown" : "reviewed";
      byId("valueError").hidden = true;
    });
  }
  function setField(key, value) {
    editSelected((d) => {
      d[key] = value;
    });
  }
  function removeBox(index) {
    const row = currentRow();
    if (!row) return;
    const work = workOf(row);
    if (index < 0 || index >= work.detections.length) return;
    work.detections.splice(index, 1);
    state.selected = -1;
    changed(row);
  }
  function removeSelected() {
    removeBox(state.selected);
  }
  function manualId(work) {
    const stamp = Date.now();
    const taken = new Set(work.detections.map((d) => d.detection_id));
    let id = "manual-" + stamp;
    for (let n = 1; taken.has(id); n++) id = "manual-" + stamp + "-" + n;
    return id;
  }
  function addManualBox(row, box, family, value) {
    const work = workOf(row);
    work.detections.push({
      bbox_xyxy: box,
      family,
      value_kph: value,
      value_state: value === null ? "unknown" : "reviewed",
      score: null,
      model_label: null,
      class_id: null,
      review_state: "pending",
      applicability: "unresolved",
      supplementary_panel: "unknown",
      detection_id: manualId(work)
    });
    state.selected = work.detections.length - 1;
    state.drawing = false;
    changed(row);
  }
  function useHint(index) {
    const row = currentRow();
    if (!row) return;
    const hint = hintsOf(row.frame_key)[index];
    if (!hint || !isBox(hint.box)) return;
    const family = familyForHint(hint.type);
    const value = familyHasValue(family) && Number.isInteger(hint.value_kph) ? hint.value_kph ?? null : null;
    addManualBox(row, [...hint.box], family, value);
  }
  function revealEditor() {
    if (narrowLayout()) return;
    byId("roads").scrollIntoView({ block: "nearest" });
  }
  function selectBox(index) {
    state.selected = index;
    state.drawing = false;
    render();
    revealEditor();
  }
  function toggleDrawing() {
    state.drawing = !state.drawing;
    state.selected = -1;
    render();
  }
  function toggleHints() {
    state.showHints = !state.showHints;
    byId("showHints").checked = state.showHints;
    writeLocal("nnslr-show-hints", state.showHints ? "1" : "0");
    render();
  }

  // web/src/review/overlay.ts
  function sceneScale(svg, width, height) {
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height || !width || !height) return 1;
    return Math.min(rect.width / width, rect.height / height);
  }
  function showLayers(visible) {
    const visibility = visible ? "" : "hidden";
    svgById("overlays").style.visibility = visibility;
    svgById("hintLayer").style.visibility = visibility;
  }
  function renderStage(row, work) {
    const scene = svgById("scene");
    scene.setAttribute("viewBox", "0 0 " + row.width + " " + row.height);
    const photo = svgById("photo");
    if (photo.getAttribute("href") !== row.preview_path) {
      photo.setAttribute("href", row.preview_path);
      byId("stage").dataset["photoKey"] = "";
      showLayers(false);
    }
    photo.setAttribute("width", String(row.width));
    photo.setAttribute("height", String(row.height));
    scene.classList.toggle("drawing", state.drawing);
    byId("stageHint").hidden = !state.drawing;
    const k = sceneScale(scene, row.width, row.height);
    const fontSize = 14 / k;
    const placed = [];
    const handleSize = 8 / k;
    const hitSize = (coarsePointer() ? 36 : 20) / k;
    const overlays = svgById("overlays");
    overlays.replaceChildren();
    const order = work.detections.map((_, index) => index);
    order.sort((a, b) => boxAreaAt(work, b) - boxAreaAt(work, a));
    for (const i of order) {
      const d = work.detections[i];
      if (d === void 0) continue;
      const b = d.bbox_xyxy;
      const isSelected = i === state.selected;
      let cls = "box";
      if (d.model_label) cls += " proposal";
      if (isSelected) cls += " selected";
      const geometry = rectOf(b);
      overlays.append(svgNode("rect", { class: cls.replace("box", "box-under"), ...geometry }));
      const rect = svgNode("rect", { class: cls, "data-index": i, ...geometry });
      if (!isSelected) rect.style.stroke = boxColor(d);
      overlays.append(rect);
      const text = i + 1 + " · " + shortName(d);
      const width = labelWidth(text, fontSize);
      const top = isSelected ? b[1] - handleSize - fontSize * 0.2 : b[1] - fontSize * 0.35;
      const y = Math.max(fontSize, freeLabelY(placed, b[0], Math.max(fontSize, top), width, fontSize, -fontSize));
      placed.push({ x: b[0], y, w: width, h: fontSize, text });
      overlays.append(label("tag", text, b[0], y, fontSize, k));
    }
    renderHintLayer(row, k, placed);
    const selected = work.detections[state.selected];
    if (selected !== void 0) appendHandles(overlays, selected.bbox_xyxy, handleSize, hitSize, true);
  }
  function boxAreaAt(work, index) {
    const d = work.detections[index];
    return d ? boxArea(d.bbox_xyxy) : 0;
  }
  function label(cls, text, x, y, fontSize, k) {
    const tag = svgNode("text", {
      x,
      y,
      class: cls,
      "font-size": fontSize,
      "stroke-width": 3.5 / k,
      "paint-order": "stroke"
    });
    tag.textContent = text;
    return tag;
  }
  var CORNERS = ["nw", "ne", "sw", "se"];
  function cornerPoint(b, corner) {
    const x = corner === "nw" || corner === "sw" ? b[0] : b[2];
    const y = corner === "nw" || corner === "ne" ? b[1] : b[3];
    return [x, y];
  }
  function appendHandles(layer, b, size, hit, outside) {
    for (const corner of CORNERS) {
      const [x, y] = cornerPoint(b, corner);
      let hx = x - size / 2;
      let hy = y - size / 2;
      if (outside) {
        hx = corner === "nw" || corner === "sw" ? x - size : x;
        hy = corner === "nw" || corner === "ne" ? y - size : y;
      }
      layer.append(svgNode("rect", { x: hx, y: hy, width: size, height: size, class: "handle" }));
    }
    for (const corner of CORNERS) {
      const [x, y] = cornerPoint(b, corner);
      layer.append(svgNode("rect", {
        x: x - hit / 2,
        y: y - hit / 2,
        width: hit,
        height: hit,
        class: "handle-hit " + corner,
        "data-corner": corner
      }));
    }
  }
  function renderHintLayer(row, k, placed) {
    const layer = svgById("hintLayer");
    layer.replaceChildren();
    if (!state.showHints) return;
    const fontSize = 12 / k;
    for (const h of hintsOf(row.frame_key)) {
      const b = h.box;
      const geometry = rectOf(b);
      layer.append(svgNode("rect", { class: "hint-under", ...geometry }));
      layer.append(svgNode("rect", { class: "hint-box " + h.confidence, ...geometry }));
      const text = hintLabel(h);
      const width = labelWidth(text, fontSize);
      if (placed.some((p) => p.text === text && p.box && boxesOverlap(p.box, b))) continue;
      const y = Math.min(row.height, freeLabelY(placed, b[0], b[3] + fontSize * 1.05, width, fontSize, fontSize));
      placed.push({ x: b[0], y, w: width, h: fontSize, text, box: b });
      layer.append(label("hint-tag", text, b[0], y, fontSize, k));
    }
  }
  function renderZoom(row, d) {
    const zoom = svgById("zoom");
    const b = d.bbox_xyxy;
    const keepView = state.drag !== null && state.drag.svg === zoom;
    if (!keepView) {
      const w = b[2] - b[0];
      const h = b[3] - b[1];
      const pad = Math.max(w, h) * 0.35 + 4;
      const x = Math.max(0, b[0] - pad);
      const y = Math.max(0, b[1] - pad);
      let vw = w + pad * 2;
      let vh = h + pad * 2;
      if (x + vw > row.width) vw = row.width - x;
      if (y + vh > row.height) vh = row.height - y;
      zoom.setAttribute("viewBox", x + " " + y + " " + vw + " " + vh);
    }
    const zoomPhoto = svgById("zoomPhoto");
    if (zoomPhoto.getAttribute("href") !== row.preview_path) zoomPhoto.setAttribute("href", row.preview_path);
    zoomPhoto.setAttribute("width", String(row.width));
    zoomPhoto.setAttribute("height", String(row.height));
    const zoomBox = svgById("zoomBox");
    for (const [name, value] of Object.entries(rectOf(b))) zoomBox.setAttribute(name, String(value));
    const view = zoom.viewBox.baseVal;
    const kz = view && view.width && view.height ? sceneScale(zoom, view.width, view.height) : 1;
    const layer = svgById("zoomHandles");
    layer.replaceChildren();
    appendHandles(layer, b, 12 / kz, (coarsePointer() ? 36 : 22) / kz, false);
  }
  function initPhoto() {
    const photo = svgById("photo");
    const loaded = (failed) => {
      const row = currentRow();
      if (row === null || photo.getAttribute("href") !== row.preview_path) return;
      byId("stage").dataset["photoKey"] = (failed ? "error:" : "") + row.frame_key;
      showLayers(true);
    };
    photo.addEventListener("load", () => loaded(false));
    photo.addEventListener("error", () => loaded(true));
  }

  // web/src/review/editor.ts
  function fill(input, value) {
    if (document.activeElement !== input) input.value = value;
  }
  function choice(className, text, pressed, onClick) {
    const button = node("button", className, text);
    button.type = "button";
    button.setAttribute("aria-pressed", pressed ? "true" : "false");
    button.addEventListener("click", onClick);
    return button;
  }
  function renderFamilies(d) {
    const usual = byId("families");
    const more = byId("familiesMore");
    usual.replaceChildren();
    more.replaceChildren();
    for (const family of FAMILIES) {
      const chip = choice("chip", family.label, d.family === family.id, () => setFamily(family.id));
      chip.dataset["family"] = family.id;
      if (family.id === "not_a_sign") {
        chip.title = "Resta come esempio negativo; per togliere un riquadro sbagliato usa ✕";
      }
      (family.minor ? more : usual).append(chip);
    }
  }
  function renderValues(d) {
    byId("valueBlock").hidden = !familyHasValue(d.family);
    const quick = byId("quickValues");
    quick.replaceChildren();
    for (const value of QUICK_VALUES) {
      const pressed = d.value_kph === value;
      const sign = choice(value >= 100 ? "sign wide" : "sign", String(value), pressed, () => setValue(value));
      sign.dataset["value"] = String(value);
      sign.setAttribute("aria-label", value + " km/h");
      quick.append(sign);
    }
    const noValue = familyHasValue(d.family) && (d.value_kph === null || d.value_kph === void 0);
    const unknown = choice("sign unknown", "?", noValue, () => setValue(null));
    unknown.dataset["value"] = "unknown";
    unknown.title = "Numero non leggibile";
    unknown.setAttribute("aria-label", "Numero non leggibile");
    quick.append(unknown);
    fill(byId("value"), d.value_kph === null || d.value_kph === void 0 ? "" : String(d.value_kph));
  }
  function renderRoads(d) {
    const roads = byId("roads");
    roads.replaceChildren();
    const current = d.applicability ? d.applicability : "unresolved";
    for (const road of ROADS) {
      const chip = choice("", road.label, current === road.id, () => setField("applicability", road.id));
      chip.dataset["applicability"] = road.id;
      roads.append(chip);
    }
  }
  function renderDetails(d) {
    for (const key of SELECT_FIELDS) fill(byId(key), d[key] ?? "unknown");
    for (const key of BOOLEAN_FIELDS) {
      const value = d[key];
      fill(byId(key), value === true ? "true" : value === false ? "false" : "unknown");
    }
    fill(byId("notes"), d.notes ?? "");
  }
  function renderEditor(row, work) {
    const d = work.detections[state.selected];
    one(".panel").classList.toggle("editing", d !== void 0);
    byId("editor").hidden = d === void 0;
    byId("remove").disabled = d === void 0;
    if (d === void 0) return;
    byId("editorTitle").textContent = "Riquadro " + (state.selected + 1);
    renderZoom(row, d);
    renderFamilies(d);
    renderValues(d);
    renderRoads(d);
    renderDetails(d);
    byId("evidence").textContent = d.model_label ? "Proposta del modello: " + d.model_label : "Riquadro aggiunto a mano.";
  }
  function renderFrameFields(row, work) {
    fill(byId("site_group"), work.site_group ?? row.site_group ?? "");
    fill(byId("encounter_id"), work.encounter_id ?? row.encounter_id ?? "");
  }
  function initEditor() {
    const valueInput = byId("value");
    valueInput.addEventListener("change", () => {
      const raw2 = valueInput.value.trim();
      if (raw2 === "") {
        setValue(null);
        return;
      }
      const value = Number(raw2);
      if (!Number.isInteger(value) || value < 5 || value > 250) {
        const error = byId("valueError");
        error.textContent = "Scrivi un numero intero tra 5 e 250, oppure lascia vuoto se non si legge.";
        error.hidden = false;
        return;
      }
      setValue(value);
    });
    for (const key of SELECT_FIELDS) {
      const select = byId(key);
      select.addEventListener("change", () => setField(key, select.value));
    }
    for (const key of BOOLEAN_FIELDS) {
      const select = byId(key);
      select.addEventListener("change", () => {
        setField(key, select.value === "true" ? true : select.value === "false" ? false : null);
      });
    }
    const notes = byId("notes");
    notes.addEventListener("change", () => setField("notes", notes.value));
    for (const key of ["site_group", "encounter_id"]) {
      const input = byId(key);
      input.addEventListener("change", () => {
        const row = currentRow();
        if (!row) return;
        const value = input.value.trim();
        workOf(row)[key] = value === "" ? null : value;
        changed(row);
      });
    }
  }

  // web/src/review/export.ts
  function say(text, ok) {
    const message = byId("exportMsg");
    message.textContent = text;
    message.className = "save-msg " + (ok ? "ok" : "error");
  }
  function exportJsonl() {
    const confirmed = rows.flatMap((row) => {
      const s = state.edits[row.frame_key];
      return s && s.review_state === "reviewed" ? [{ row, s }] : [];
    });
    if (confirmed.some(({ s }) => !s.reviewer || !s.review_timestamp)) {
      say("Riconferma le vecchie revisioni per registrarne revisore e data.", false);
      return;
    }
    const lines = confirmed.map(({ row, s }) => {
      const line = {
        ...row,
        reviewer: s.reviewer,
        review_timestamp: s.review_timestamp,
        site_group: s.site_group ?? row.site_group ?? null,
        encounter_id: s.encounter_id ?? row.encounter_id ?? null,
        kind: "reviewed_frame",
        ground_truth: false,
        training_ready: false,
        review_state: "reviewed",
        label_basis: "human_review",
        source_proposals: row.detections,
        detections: s.detections.map((d) => ({
          ...d,
          review_state: "reviewed",
          value_state: d.value_kph == null ? "unknown" : "reviewed"
        }))
      };
      if (s.reviewer_kind === "ai") {
        line.label_basis = "ai_review";
        line.reviewer_kind = "ai";
        line.review_channel = s.channel;
      }
      return JSON.stringify(line);
    });
    if (lines.length === 0) {
      say("Conferma almeno un fotogramma prima di esportare.", false);
      return;
    }
    const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "application/x-ndjson" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "annotations-" + runId + ".jsonl";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1e3);
    say(lines.length + (lines.length === 1 ? " fotogramma esportato." : " fotogrammi esportati."), true);
  }

  // web/src/review/keyboard.ts
  var CONFIRM_ON_ENTER = /* @__PURE__ */ new Set(["prev", "next", "mPrev", "mNext", "draw"]);
  function initKeyboard() {
    let pointerButton = null;
    document.addEventListener("pointerdown", (event) => {
      pointerButton = event.target instanceof Element ? event.target.closest("button") : null;
    }, true);
    document.addEventListener("focusin", (event) => {
      if (event.target !== pointerButton) pointerButton = null;
    });
    document.addEventListener("keydown", (event) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      const tag = target ? target.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "select" || tag === "textarea") {
        if (event.key === "Escape") target?.blur();
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const byPointer = target !== null && target === pointerButton;
      if (event.key === "Enter" && byPointer && CONFIRM_ON_ENTER.has(target.id)) {
        event.preventDefault();
        void confirmFrame();
        return;
      }
      if (event.key === "Enter" && (tag === "button" || tag === "summary" || tag === "a")) return;
      switch (event.key) {
        case "ArrowLeft":
          event.preventDefault();
          go(-1);
          break;
        case "ArrowRight":
          event.preventDefault();
          go(1);
          break;
        case "Enter":
          event.preventDefault();
          void confirmFrame();
          break;
        case "n":
        case "N":
          event.preventDefault();
          toggleDrawing();
          break;
        case "h":
        case "H":
          if (routeHasHints) {
            event.preventDefault();
            toggleHints();
          }
          break;
        case "Delete":
        case "Backspace":
          if (state.selected >= 0) {
            event.preventDefault();
            removeSelected();
          }
          break;
        case "Escape":
          state.drawing = false;
          state.selected = -1;
          render();
          break;
      }
    });
  }

  // web/src/review/layout.ts
  function renderOverlaysOnly() {
    const row = currentRow();
    if (!row || byId("workspace").hidden || state.drag) return;
    renderStage(row, workOf(row));
    const d = selectedDetection();
    if (d) renderZoom(row, d);
  }
  function measure(element, property) {
    document.documentElement.style.setProperty(property, Math.ceil(element.getBoundingClientRect().height) + "px");
  }
  function initLayout() {
    const topbar = one(".topbar");
    const mobilebar = byId("mobilebar");
    if (typeof ResizeObserver === "function") {
      const overlays = new ResizeObserver(() => renderOverlaysOnly());
      overlays.observe(byId("stage"));
      overlays.observe(svgById("zoom"));
      new ResizeObserver(() => measure(topbar, "--topbar-h")).observe(topbar);
      new ResizeObserver(() => measure(mobilebar, "--mobilebar-h")).observe(mobilebar);
    } else {
      window.addEventListener("resize", () => {
        renderOverlaysOnly();
        measure(topbar, "--topbar-h");
        measure(mobilebar, "--mobilebar-h");
      });
      measure(topbar, "--topbar-h");
      measure(mobilebar, "--mobilebar-h");
    }
  }

  // web/src/review/pointer.ts
  function isCorner(value) {
    return value === "nw" || value === "ne" || value === "sw" || value === "se";
  }
  function cornerOf(target) {
    const corner = datasetOf(target)?.["corner"];
    return isCorner(corner) ? corner : null;
  }
  function toImagePoint(svg, event, row) {
    const matrix = svg.getScreenCTM();
    let point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    if (matrix !== null) point = point.matrixTransform(matrix.inverse());
    return [Math.min(Math.max(point.x, 0), row.width), Math.min(Math.max(point.y, 0), row.height)];
  }
  function drawDraft(a, b) {
    const draft = svgById("draft");
    draft.setAttribute("x", String(Math.min(a[0], b[0])));
    draft.setAttribute("y", String(Math.min(a[1], b[1])));
    draft.setAttribute("width", String(Math.abs(a[0] - b[0])));
    draft.setAttribute("height", String(Math.abs(a[1] - b[1])));
  }
  function onPointerDown(event) {
    const row = currentRow();
    const svg = event.currentTarget;
    if (!row || !(svg instanceof SVGSVGElement)) return;
    if (svg.id === "scene" && state.drawing) {
      event.preventDefault();
      const start2 = toImagePoint(svg, event, row);
      state.drag = { kind: "draw", svg, start: start2 };
      svg.setPointerCapture(event.pointerId);
      svgById("draft").setAttribute("visibility", "visible");
      drawDraft(start2, start2);
      return;
    }
    const corner = cornerOf(event.target);
    if (corner !== null) {
      event.preventDefault();
      state.drag = { kind: "resize", svg, corner, moved: false };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("dragging");
    }
  }
  function onPointerMove(event) {
    const drag = state.drag;
    if (!drag || event.currentTarget !== drag.svg) return;
    const row = currentRow();
    if (!row) return;
    const p = toImagePoint(drag.svg, event, row);
    if (drag.kind === "draw") {
      drawDraft(drag.start, p);
      return;
    }
    const d = selectedDetection();
    if (!d) return;
    const b = [...d.bbox_xyxy];
    if (drag.corner === "nw") {
      b[0] = p[0];
      b[1] = p[1];
    }
    if (drag.corner === "ne") {
      b[2] = p[0];
      b[1] = p[1];
    }
    if (drag.corner === "sw") {
      b[0] = p[0];
      b[3] = p[1];
    }
    if (drag.corner === "se") {
      b[2] = p[0];
      b[3] = p[1];
    }
    const x1 = Math.min(b[0], b[2]);
    const x2 = Math.max(b[0], b[2]);
    const y1 = Math.min(b[1], b[3]);
    const y2 = Math.max(b[1], b[3]);
    if (x2 - x1 >= 2 && y2 - y1 >= 2) {
      d.bbox_xyxy = [x1, y1, x2, y2];
      drag.moved = true;
      renderStage(row, workOf(row));
      renderZoom(row, d);
    }
  }
  function finishDrag(event) {
    const row = currentRow();
    const drag = state.drag;
    if (!row || !drag || event.currentTarget !== drag.svg) return;
    state.drag = null;
    drag.svg.classList.remove("dragging");
    state.ignoreClickUntil = Date.now() + 400;
    if (drag.kind === "draw") {
      svgById("draft").setAttribute("visibility", "hidden");
      if (event.type === "pointercancel") return;
      const p = toImagePoint(drag.svg, event, row);
      const box = [
        Math.min(drag.start[0], p[0]),
        Math.min(drag.start[1], p[1]),
        Math.max(drag.start[0], p[0]),
        Math.max(drag.start[1], p[1])
      ];
      if (box[2] - box[0] < 2 || box[3] - box[1] < 2) return;
      addManualBox(row, box, "unreadable", null);
      return;
    }
    if (drag.moved) changed(row);
    else render();
  }
  function onTouchStart(event) {
    const svg = event.currentTarget;
    if (svg instanceof SVGSVGElement && svg.id === "scene" && state.drawing) {
      event.preventDefault();
      return;
    }
    if (cornerOf(event.target) !== null) event.preventDefault();
  }
  function onSceneClick(event) {
    if (Date.now() < state.ignoreClickUntil || state.drawing) return;
    const data = datasetOf(event.target);
    const index = data?.["index"];
    if (index !== void 0) {
      state.selected = Number(index);
      render();
      revealEditor();
      return;
    }
    if (data?.["corner"]) return;
    if (state.selected !== -1) {
      state.selected = -1;
      render();
    }
  }
  function initPointer() {
    for (const svg of [svgById("scene"), svgById("zoom")]) {
      svg.addEventListener("pointerdown", onPointerDown);
      svg.addEventListener("pointermove", onPointerMove);
      svg.addEventListener("pointerup", finishDrag);
      svg.addEventListener("pointercancel", finishDrag);
      svg.addEventListener("touchstart", onTouchStart, { passive: false });
    }
    svgById("scene").addEventListener("click", onSceneClick);
  }

  // web/src/review/boxes.ts
  function originalOf(row, d) {
    if (!d.detection_id) return null;
    return proposalsOf(row).find((p) => p.detection_id === d.detection_id) ?? null;
  }
  function sourceLine(row, d) {
    const src = node("span", "src");
    const original = originalOf(row, d);
    if (original === null) {
      src.textContent = "aggiunto a mano";
    } else if (original.family !== d.family || original.value_kph !== d.value_kph) {
      src.append(
        document.createTextNode("modello " + shortName(original) + " "),
        node("span", "changed", "→ " + shortName(d))
      );
    } else if (original.score === null || original.score === void 0) {
      src.textContent = "modello";
    } else {
      src.textContent = "modello · " + Math.round(Number(original.score) * 100) + "%";
    }
    return src;
  }
  function renderBoxList(row, work) {
    const list = byId("boxList");
    list.replaceChildren();
    work.detections.forEach((d, i) => {
      const selected = i === state.selected;
      const item = node("div", selected ? "box-row selected" : "box-row");
      item.dataset["index"] = String(i);
      item.dataset["detectionId"] = String(d.detection_id || "");
      const main = node("button", "box-main");
      main.type = "button";
      main.setAttribute("aria-pressed", selected ? "true" : "false");
      const src = sourceLine(row, d);
      const earlier = work.detections.slice(0, i).findIndex((other) => overlap(other.bbox_xyxy, d.bbox_xyxy) > 0.6);
      if (earlier >= 0) {
        src.append(node("span", "dup", " · stesso cartello di " + (earlier + 1)));
        item.dataset["duplicateOf"] = String(earlier);
      }
      main.append(node("span", "n", String(i + 1)), node("span", "what", describe(d)), src);
      main.addEventListener("click", () => selectBox(i));
      const remove = node("button", "box-del", "✕");
      remove.type = "button";
      remove.title = "Elimina il riquadro " + (i + 1);
      remove.setAttribute("aria-label", "Elimina il riquadro " + (i + 1));
      remove.addEventListener("click", () => removeBox(i));
      item.append(main, remove);
      list.append(item);
    });
    const count = work.detections.length;
    byId("noBoxes").hidden = count > 0;
    byId("pickHint").hidden = count === 0 || state.selected >= 0;
    byId("boxCount").textContent = count === 0 ? "" : count === 1 ? "1 riquadro" : count + " riquadri";
  }

  // web/src/review/hints.ts
  function groupReadings(hints) {
    const groups = [];
    hints.forEach((hint, i) => {
      let found = null;
      for (const group of groups) {
        const first = hints[group[0] ?? -1];
        const same = first && first.type === hint.type && first.value_kph === hint.value_kph;
        if (same && overlap(first.box, hint.box) > 0.3) found = group;
      }
      if (found === null) groups.push([i]);
      else found.push(i);
    });
    return groups;
  }
  function alreadyUsed(work, h) {
    return work.detections.some((d) => sameBox(d.bbox_xyxy, h.box) || overlap(d.bbox_xyxy, h.box) > 0.3 && d.family === familyForHint(h.type) && d.value_kph === h.value_kph);
  }
  function hintRow(hints, group, work) {
    const i = group[0];
    const h = i === void 0 ? void 0 : hints[i];
    if (i === void 0 || h === void 0) return null;
    const item = node("div", "hint-row");
    const body = node("div", "hint-text");
    const head = node("div", "hint-head");
    let text = hintTypeOf(h.type).label;
    if (hasValue(h)) text += " · " + h.value_kph + " km/h";
    head.append(node("span", "hint-what", text), node("span", "conf " + h.confidence, confidenceLabel(h.confidence)));
    const passes = [];
    for (const k of group) {
      const pass = hints[k]?.pass;
      if (pass && !passes.includes(pass)) passes.push(pass);
    }
    const parts = [];
    if (passes.length === 1) parts.push("lettura " + passes[0]);
    if (passes.length > 1) parts.push("letture " + passes.join(" e "));
    if (h.side) parts.push(sideLabel(h.side));
    if (parts.length > 0) head.append(node("span", "hint-sub", parts.join(" · ")));
    body.append(head);
    if (h.note) body.append(node("p", "hint-note", h.note));
    item.append(body);
    const use = node("button", "btn small tint");
    use.type = "button";
    use.dataset["hintIndex"] = String(i);
    if (alreadyUsed(work, h)) {
      use.textContent = "Già usato";
      use.disabled = true;
    } else {
      use.textContent = "Usa";
      use.setAttribute("aria-label", "Usa questo: " + text + (h.pass ? " (lettura " + h.pass + ")" : ""));
      use.addEventListener("click", () => useHint(i));
    }
    item.append(use);
    return item;
  }
  function statsLine() {
    if (!hintStats) return "";
    let stats = "Seconda lettura su " + hintStats.frames + (hintStats.frames === 1 ? " fotogramma" : " fotogrammi") + " di questa route";
    const skipped = (hintStats.skipped_lines || 0) + (hintStats.skipped_signs || 0) + (hintStats.skipped_files || 0);
    if (skipped === 1) stats += " · 1 voce scartata perché non valida";
    if (skipped > 1) stats += " · " + skipped + " voci scartate perché non valide";
    return stats + ".";
  }
  function renderHintList(row, work) {
    const hints = hintsOf(row.frame_key);
    byId("hintToggle").hidden = !routeHasHints;
    byId("keyHints").hidden = !routeHasHints;
    byId("hintSection").hidden = hints.length === 0;
    if (hints.length === 0) return;
    const rowsOfHints = groupReadings(hints).map((group) => hintRow(hints, group, work));
    byId("hintList").replaceChildren(...rowsOfHints.filter((item) => item !== null));
    byId("hintStats").textContent = statsLine();
  }

  // web/src/review/messages.ts
  function noticeNode(text, className) {
    const box = node("div", className);
    box.setAttribute("role", "alert");
    box.append(node("span", "", text));
    return box;
  }
  function renderNotices() {
    const box = byId("notices");
    box.replaceChildren();
    if (state.serverWarning) box.append(noticeNode(state.serverWarning, "notice"));
    if (state.storageWarning) box.append(noticeNode(state.storageWarning, "notice"));
    const unsaved = apiBase ? unsavedKeys().filter((key) => !(state.inFlight.has(key) && !state.edits[key]?.error)) : [];
    if (unsaved.length === 0) return;
    let list = unsaved.slice(0, 8).join(", ");
    if (unsaved.length > 8) list += ", …";
    const text = unsaved.length === 1 ? "1 fotogramma confermato NON salvato sul server (" + list + ")." : unsaved.length + " fotogrammi confermati NON salvati sul server (" + list + ").";
    const notice = noticeNode(text, "notice error");
    const retry = node("button", "btn small", "Riprova a salvare");
    retry.type = "button";
    retry.addEventListener("click", () => void retryUnsaved(true));
    notice.append(retry);
    box.append(notice);
  }
  function showState(title, paragraphs) {
    const box = byId("stateBox");
    box.replaceChildren(node("h2", "", title), ...paragraphs.map((text) => node("p", "", text)));
    box.hidden = false;
    byId("workspace").hidden = true;
    byId("mobilebar").hidden = true;
  }
  function saveText(key, s) {
    const status = frameStatus(key);
    if (!apiBase) {
      if (status === "done") return ["Confermato in questo browser: esporta il JSONL per conservarlo.", "ok"];
      if (status === "dirty") return ["Modifiche non confermate (solo in questo browser).", "warn"];
      return ["", ""];
    }
    if (status === "saving") return ["Salvataggio in corso…", ""];
    if (!s) return ["", ""];
    if (status === "done" && aiConfirmed(key)) {
      const via = s.channel === "mcp" ? "MCP" : s.channel === "rest" ? "REST" : "";
      const who = s.reviewer || "un’AI";
      return ["Salvato ✓ · revisione " + s.revision + " · " + who + " (AI" + (via ? " · " + via : "") + ")", "ok"];
    }
    if (status === "done") {
      return ["Salvato ✓ · revisione " + s.revision + (s.reviewer ? " · " + s.reviewer : ""), "ok"];
    }
    if (status === "error" && s.error) return ["NON salvato: " + s.error, "error"];
    if (status === "error") return ["Confermato ma non ancora salvato: nuovo tentativo automatico.", "error"];
    if (status === "dirty" && s.server_confirmed) {
      return ["Modificato: premi Conferma per salvarlo. Sul server resta la versione di prima.", "warn"];
    }
    if (status === "dirty") return ["Modifiche da confermare.", "warn"];
    if (s.base && s.base.confirmed === false) {
      return ["Conferma annullata: non verrà importato finché non lo confermi di nuovo.", ""];
    }
    return ["", ""];
  }
  function renderConfirmButtons(key, work, status) {
    const names = work.detections.map(shortName);
    let label2 = "Conferma · " + names.join(" · ");
    let mobileLabel = label2;
    if (work.detections.length === 0) {
      label2 = "Conferma · nessun cartello";
      mobileLabel = "Nessun cartello";
    }
    let already = false;
    if (status === "done" && apiBase && sameAsServer(state.edits[key])) {
      label2 = "Già confermato · avanti →";
      mobileLabel = "Avanti →";
      already = true;
    } else if (status === "done") {
      label2 = "Confermato ✓ · conferma di nuovo";
      mobileLabel = "Confermato ✓";
    }
    if (status === "saving") {
      label2 = "Salvataggio…";
      mobileLabel = "Salvataggio…";
    }
    byId("confirmLabel").textContent = label2;
    byId("mConfirmLabel").textContent = mobileLabel;
    for (const id of ["confirm", "mConfirm"]) {
      const button = byId(id);
      button.disabled = status === "saving";
      button.dataset["saveState"] = status;
      button.classList.toggle("already", already);
    }
  }
  function renderConfirmArea(key, work, status) {
    renderConfirmButtons(key, work, status);
    const s = state.edits[key];
    const discard = byId("discard");
    const withdraw = byId("withdraw");
    discard.hidden = !(status === "dirty" || status === "error");
    withdraw.hidden = !(apiBase && s && s.saved === true && s.server_confirmed === true && status !== "saving");
    const armed = state.withdrawArmed === key;
    withdraw.textContent = armed ? "Sì, annulla" : "Annulla conferma";
    withdraw.dataset["armed"] = armed ? "true" : "false";
    const [savedText, savedClass] = saveText(key, s);
    const notice = state.notice && state.notice.key === key ? state.notice : null;
    let lineText = savedText;
    let lineClass = savedClass;
    if (notice !== null) {
      if (notice.error) {
        lineText = notice.text;
        lineClass = "error";
      } else if (savedClass !== "error") {
        lineText = notice.text;
        lineClass = notice.warn ? "warn" : "ok";
      }
    }
    const line = byId("saveMsg");
    line.textContent = lineText;
    line.title = lineText;
    line.className = "save-msg" + (lineClass ? " " + lineClass : "");
    let mobileText = "";
    const unsaved = apiBase ? unsavedKeys().length : 0;
    if (notice && notice.error) mobileText = notice.text;
    else if (savedClass === "error") mobileText = savedText;
    else if (state.serverWarning) mobileText = state.serverWarning;
    else if (state.storageWarning) mobileText = state.storageWarning;
    else if (unsaved > 0) mobileText = unsaved + " fotogrammi confermati NON salvati: riprova dal pannello.";
    const mobile = byId("mobileMsg");
    mobile.textContent = mobileText;
    mobile.hidden = mobileText === "";
    const quiet = lineText === "" && byId("notices").childElementCount === 0 && discard.hidden && withdraw.hidden;
    one(".confirm-section").classList.toggle("quiet", quiet);
  }

  // web/src/review/topbar.ts
  function renderRoutes() {
    const select = byId("route");
    select.replaceChildren();
    let found = false;
    for (const route of routes) {
      const option = node("option", "", String(route.route_id || ""));
      option.value = String(route.url || "");
      if (route.route_id === routeId) {
        option.selected = true;
        found = true;
      }
      select.append(option);
    }
    if (!found) {
      const option = node("option", "", routeId ? routeId : "Report locale");
      option.value = "";
      option.selected = true;
      select.prepend(option);
    }
    select.disabled = routes.length <= 1;
  }
  function renderProgress() {
    const total = rows.length;
    let done = 0;
    let suspicious = 0;
    let byAi = 0;
    for (const row of rows) {
      if (isConfirmed(row.frame_key)) done += 1;
      if (reasonsOf(row.frame_key).length > 0) suspicious += 1;
      if (aiConfirmed(row.frame_key)) byAi += 1;
    }
    const aiButton = one('.filters button[data-filter="ai"]');
    aiButton.textContent = "AI · " + byAi;
    aiButton.hidden = byAi === 0 && state.filter !== "ai";
    let summary = done + "/" + total + " confermati";
    if (byAi > 0) summary += " · " + byAi + " AI";
    if (suspicious > 0) summary += " · " + suspicious + (suspicious === 1 ? " sospetto" : " sospetti");
    byId("tlSub").textContent = summary;
    byId("progressShort").textContent = done + "/" + total;
    const text = byId("progressText");
    text.replaceChildren();
    if (total === 0) {
      text.textContent = "Nessun fotogramma";
    } else {
      text.append(node("strong", "", done + " di " + total), document.createTextNode(" confermati"));
      if (suspicious > 0 || routeHasHints) {
        const sus = node("span", suspicious > 0 ? "sus" : "", suspicious === 1 ? "1 sospetto" : suspicious + " sospetti");
        text.append(document.createTextNode(" · "), sus);
      }
    }
    byId("progressBar").style.width = total === 0 ? "0" : Math.round(done * 100 / total) + "%";
    renderQuality();
  }
  function markFilters() {
    for (const button of document.querySelectorAll(".filters button")) {
      button.setAttribute("aria-pressed", button.dataset["filter"] === state.filter ? "true" : "false");
    }
  }
  function initTopbar() {
    for (const button of document.querySelectorAll(".filters button, .qchip")) {
      button.addEventListener("click", () => {
        const filter = button.dataset["filter"];
        if (!isFilter(filter)) return;
        if (isQualityFilter(filter) && state.filter === filter) setFilter("todo");
        else setFilter(filter);
      });
    }
    const select = byId("route");
    select.addEventListener("change", () => {
      if (select.value.startsWith("/route/")) window.location.assign(select.value);
    });
    const reviewer = byId("reviewer");
    reviewer.addEventListener("input", () => {
      reviewer.classList.remove("need");
      writeLocal("nnslr-reviewer", reviewer.value.trim());
    });
  }

  // web/src/review/render.ts
  function renderFrameBar(row, status) {
    const key = row.frame_key;
    const list = visibleRows();
    const pos = Math.max(0, list.findIndex((item) => item.frame_key === key));
    byId("pos").textContent = pos + 1 + " / " + list.length;
    for (const id of ["prev", "mPrev"]) byId(id).disabled = pos <= 0;
    for (const id of ["next", "mNext"]) byId(id).disabled = pos >= list.length - 1;
    const reasons = reasonsOf(key);
    const pill = byId("statePill");
    pill.className = "pill state " + status + (aiConfirmed(key) ? " ai" : "") + (status === "todo" && reasons.length > 0 ? " sus" : "");
    pill.textContent = statusLabel(key);
    const proposals = proposalsOf(row).length;
    let info = "Segmento " + row.segment_index + " · immagine " + row.output_index + " · " + proposals + (proposals === 1 ? " proposta" : " proposte");
    if (routeHasHints) info += " · " + hintsOf(key).length + " in seconda lettura";
    byId("frameData").textContent = info;
    const reasonBox = byId("reasons");
    reasonBox.replaceChildren();
    if (reasons.length > 0) {
      const strong = document.createElement("strong");
      strong.textContent = "Sospetto: ";
      reasonBox.append(strong, document.createTextNode(reasons.map((reason) => String(reason.text)).join("; ")));
    }
    reasonBox.hidden = reasons.length === 0;
    const note = byId("filterNote");
    note.textContent = state.filterNote;
    note.hidden = state.filterNote === "";
    const draw = byId("draw");
    draw.classList.toggle("on", state.drawing);
    draw.textContent = state.drawing ? "Annulla" : "+ Riquadro";
  }
  function publishState(key, status, s) {
    const data = document.body.dataset;
    data["frameKey"] = key;
    data["frameStatus"] = status;
    data["frameReviewerKind"] = s && s.server_confirmed ? s.reviewer_kind || "human" : "";
    data["server"] = state.server;
    data["unsaved"] = String(unsavedKeys().length);
    data["saving"] = state.inFlight.size > 0 ? "true" : "false";
    data["saveCount"] = String(state.saveCount);
    if (state.lastSave) {
      data["lastSaveKey"] = state.lastSave.key;
      data["lastSaveResult"] = state.lastSave.result;
      data["lastSaveRevision"] = String(state.lastSave.revision);
    }
  }
  function renderPage() {
    renderNotices();
    renderProgress();
    markFilters();
    let row = currentRow();
    if (row === null) {
      row = rows[0] ?? null;
      if (row === null) {
        showState(
          "Questo report non ha fotogrammi",
          ["Scegli un’altra route in alto, oppure rigenera la pre-annotazione."]
        );
        return;
      }
      state.currentKey = row.frame_key;
    }
    byId("stateBox").hidden = true;
    byId("workspace").hidden = false;
    byId("mobilebar").hidden = false;
    const key = row.frame_key;
    const work = workOf(row);
    if (state.selected >= work.detections.length) state.selected = -1;
    renderStage(row, work);
    renderBoxList(row, work);
    renderEditor(row, work);
    renderHintList(row, work);
    renderTimeline();
    const status = frameStatus(key);
    renderFrameBar(row, status);
    renderConfirmArea(key, work, status);
    renderFrameFields(row, work);
    publishState(key, status, state.edits[key]);
  }

  // web/src/pages/review.ts
  var SERVER_NOTE = "Ogni conferma viene salvata nel database del server. Se il salvataggio non riesce, il fotogramma resta segnato “non salvato” e viene ritentato; nel frattempo le modifiche restano in questo browser. Il JSONL resta disponibile come copia di sicurezza. Un frame senza riquadri non è automaticamente un esempio negativo: va controllato e confermato.";
  var OFFLINE_NOTE = "Le modifiche sono salvate in questo browser. Esporta il JSONL per conservarle fuori dal browser; il file originale delle proposte resta invariato. Un frame senza riquadri non è automaticamente un esempio negativo: va controllato e confermato.";
  function initControls() {
    const click = (id, action) => byId(id).addEventListener("click", action);
    click("prev", () => go(-1));
    click("next", () => go(1));
    click("mPrev", () => go(-1));
    click("mNext", () => go(1));
    click("confirm", () => void confirmFrame());
    click("mConfirm", () => void confirmFrame());
    click("withdraw", () => void withdrawFrame());
    click("discard", discardChanges);
    click("export", exportJsonl);
    click("draw", toggleDrawing);
    click("remove", removeSelected);
    const showHints = byId("showHints");
    showHints.addEventListener("change", () => {
      if (showHints.checked !== state.showHints) toggleHints();
    });
  }
  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
      if (!saved || typeof saved !== "object" || Array.isArray(saved)) throw new Error("Invalid review storage");
      state.edits = saved;
    } catch {
      state.storageWarning = "Revisioni precedenti non recuperabili dal browser. Esporta il JSONL prima di chiudere.";
    }
    byId("reviewer").value = readLocal("nnslr-reviewer", "") || "";
    const savedFilter = readLocal("nnslr-review-filter", "todo");
    if (isFilter(savedFilter)) state.filter = savedFilter;
    if (readLocal("nnslr-timeline-open", "0") === "1") setTimelineOpen(true);
    state.showHints = readLocal("nnslr-show-hints", "1") !== "0";
    byId("showHints").checked = state.showHints;
  }
  function start() {
    setRenderer(renderPage);
    initThemeSwitch();
    initTopbar();
    initControls();
    initEditor();
    initPointer();
    initPhoto();
    initTimeline(openFrame);
    initKeyboard();
    initLayout();
    restore();
    if (apiBase) {
      byId("storageNote").textContent = SERVER_NOTE;
    } else {
      const brand = byId("brand");
      brand.removeAttribute("href");
      brand.removeAttribute("title");
      one(".account-btn").hidden = true;
      byId("storageNote").textContent = OFFLINE_NOTE;
    }
    renderRoutes();
    buildTimeline();
    state.currentKey = pickStartFrame();
    if (apiBase) state.server = "loading";
    render();
    scrollTimelineToCurrent();
    if (apiBase) {
      void loadServer();
      window.setInterval(() => void retryUnsaved(false), 3e4);
      window.addEventListener("online", () => void retryUnsaved(false));
    }
  }
  start();
})();
