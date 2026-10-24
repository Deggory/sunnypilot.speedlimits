"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

  // web/src/common/api.ts
  var ApiError = class extends Error {
    constructor(message, status2) {
      super(message);
      __publicField(this, "status", status2);
    }
  };
  async function api(path, options = {}, signIn = false) {
    const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options });
    if (response.status === 401 && !signIn) {
      location.replace("/");
      throw new ApiError("Sessione finita.", 401);
    }
    let data;
    try {
      data = await response.json();
    } catch {
      data = { message: "Risposta non valida dal server (HTTP " + response.status + ")." };
    }
    if (!response.ok) {
      const problem = data;
      throw new ApiError(problem.message || "HTTP " + response.status, response.status);
    }
    return data;
  }
  function post(path, body = {}, signIn = false) {
    return api(
      path,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
      signIn
    );
  }
  function messageOf(error) {
    return error instanceof Error ? error.message : String(error);
  }

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
  function pill(label, kind) {
    return node("span", "pill " + kind, label);
  }
  function setBar(id, percent, kind = "") {
    const bar = byId(id);
    bar.style.width = Math.max(0, Math.min(100, percent)) + "%";
    if (bar.parentElement) bar.parentElement.className = "bar" + (kind ? " " + kind : "");
  }
  var shown = /* @__PURE__ */ new Map();
  function unchanged(key, value) {
    const text = JSON.stringify(value);
    if (shown.get(key) === text) return true;
    shown.set(key, text);
    return false;
  }

  // web/src/common/format.ts
  function clock() {
    return (/* @__PURE__ */ new Date()).toLocaleTimeString("it-IT");
  }

  // web/src/common/theme.ts
  var KEY = "nnslr-theme";
  var CHANGED = "nnslr-theme";
  var root = document.documentElement;
  function storedTheme() {
    try {
      const value = localStorage.getItem(KEY);
      if (value === "light" || value === "auto") return value;
    } catch {
    }
    return "dark";
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

  // web/src/common/account-menu.ts
  var THEMES = [["dark", "Scuro"], ["light", "Chiaro"], ["auto", "Automatico"]];
  function themeChoice() {
    const group = node("div", "segmented");
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-label", "Aspetto");
    const buttons = THEMES.map(([theme, label]) => {
      const button = node("button", "", label);
      button.type = "button";
      button.setAttribute("role", "radio");
      button.addEventListener("click", () => setTheme(theme));
      return { theme, button };
    });
    const show = () => {
      const current = storedTheme();
      for (const { theme, button } of buttons) button.setAttribute("aria-checked", String(theme === current));
    };
    onThemeChange(show);
    show();
    group.append(...buttons.map(({ button }) => button));
    return group;
  }
  function who(state, error) {
    const head = node("div", "pop-head");
    const avatar = node("span", "pop-avatar", (state?.user ?? "?").slice(0, 1).toUpperCase());
    avatar.setAttribute("aria-hidden", "true");
    const text = node("div", "pop-who");
    if (state === null) {
      text.append(node("strong", "", "Account"), node("span", "", error || "Caricamento…"));
    } else if (state.local) {
      text.append(node("strong", "", "Questo computer"), node("span", "", "Sito su 127.0.0.1, senza accesso"));
    } else {
      const sessions = state.sessions ?? 1;
      text.append(
        node("strong", "", state.user ?? ""),
        node("span", "", sessions > 1 ? "Connesso da " + sessions + " browser" : "Connesso da questo browser")
      );
    }
    head.append(avatar, text);
    return head;
  }
  async function signOut(everywhere) {
    if (everywhere && !window.confirm("Uscire da tutti i browser, compreso questo?")) return;
    try {
      await post(everywhere ? "/api/auth/logout-all" : "/api/auth/logout");
    } catch {
    }
    location.replace("/");
  }
  function menuButton(label, className, action) {
    const button = node("button", "pop-item " + className, label);
    button.type = "button";
    button.addEventListener("click", action);
    return button;
  }
  function fill(pop, state, error = "") {
    const appearance = node("div", "pop-section");
    appearance.append(node("div", "pop-label", "Aspetto"), themeChoice());
    const items = node("div", "pop-section pop-items");
    const settings = node("a", "pop-item", "Password e browser connessi");
    settings.href = "/account";
    items.appendChild(settings);
    if (state !== null && !state.local) {
      items.append(
        menuButton("Esci", "", () => void signOut(false)),
        menuButton("Esci da tutti i browser", "danger", () => void signOut(true))
      );
    }
    pop.replaceChildren(who(state, error), appearance, items);
  }
  function initAccountMenu() {
    const button = document.getElementById("account-btn");
    const pop = document.getElementById("account-pop");
    if (button === null || pop === null) return;
    let state = null;
    const open = () => {
      fill(pop, state);
      pop.hidden = false;
      button.setAttribute("aria-expanded", "true");
      api("/api/auth/state").then((answer) => {
        state = answer;
        if (!pop.hidden) fill(pop, state);
      }).catch((problem) => {
        if (!pop.hidden && state === null) fill(pop, null, problem instanceof Error ? problem.message : String(problem));
      });
      pop.querySelector('[aria-checked="true"]')?.focus();
    };
    const close = (refocus) => {
      if (pop.hidden) return;
      pop.hidden = true;
      button.setAttribute("aria-expanded", "false");
      if (refocus) button.focus();
    };
    button.addEventListener("click", () => pop.hidden ? open() : close(false));
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!pop.contains(target) && !button.contains(target)) close(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !pop.hidden) {
        event.stopPropagation();
        close(true);
      }
    }, true);
  }

  // web/src/common/toast.ts
  var timer;
  function toast(message, error = false) {
    const element = byId("toast");
    element.textContent = message;
    element.className = "toast show" + (error ? " error" : "");
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      element.className = "toast";
    }, error ? 7e3 : 3500);
  }

  // web/src/pages/status.ts
  var JOB_STATE = {
    running: ["in corso", "run"],
    ok: ["finito", "ok"],
    failed: ["fallito", "bad"],
    interrupted: ["interrotto", "warn"]
  };
  var HISTORY_POINTS = 90;
  var usage = { cpu: [], gpu: [] };
  var status = null;
  var log = { id: null, offset: 0, timer: void 0, follow: true };
  function pushUsage(key, value) {
    const history = usage[key];
    history.push(value);
    if (history.length > HISTORY_POINTS) history.shift();
  }
  function renderSpark(id, values, color) {
    const SVG = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 " + HISTORY_POINTS + " 30");
    svg.setAttribute("preserveAspectRatio", "none");
    if (values.length > 1) {
      const start = HISTORY_POINTS - values.length;
      const points = values.map((value, index) => start + index + "," + (30 - Math.max(0, Math.min(100, value)) * 0.3));
      const area = document.createElementNS(SVG, "polygon");
      area.setAttribute("points", start + ",30 " + points.join(" ") + " " + (HISTORY_POINTS - 1) + ",30");
      area.style.fill = color;
      area.setAttribute("fill-opacity", "0.15");
      const line = document.createElementNS(SVG, "polyline");
      line.setAttribute("points", points.join(" "));
      line.setAttribute("fill", "none");
      line.style.stroke = color;
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("vector-effect", "non-scaling-stroke");
      svg.append(area, line);
    }
    byId(id).replaceChildren(svg);
  }
  function renderCores(cores) {
    const box = byId("cpu-cores");
    if (box.children.length !== cores.length) {
      box.replaceChildren(...cores.map(() => node("span", "core")));
    }
    cores.forEach((value, index) => {
      const bar = box.children[index];
      bar.style.setProperty("--h", Math.max(4, value) + "%");
      bar.className = "core" + (value >= 85 ? " hot" : value >= 50 ? " warm" : "");
      bar.title = "core " + index + ": " + Math.round(value) + "%";
    });
  }
  function statRows(id, rows) {
    if (unchanged("stats:" + id, rows)) return;
    byId(id).replaceChildren(...rows.flatMap(([label, value]) => [
      node("span", "stat-label", label),
      node("span", "stat-value", value)
    ]));
  }
  function renderMachine(current) {
    const system = current.system;
    const gpu = system.gpu;
    if (gpu) {
      const util = gpu.utilization ?? 0;
      byId("gpu-value").textContent = util + "%";
      setBar("gpu-bar", util, util > 85 ? "warn" : "");
      pushUsage("gpu", util);
      renderSpark("gpu-spark", usage.gpu, "var(--spark-gpu)");
      const busy = gpu.process_list.length || gpu.processes;
      const rows = [["Scheda", gpu.name.replace(/^Tesla /, "")]];
      if (gpu.memory_total_mb && gpu.memory_used_mb !== null) {
        rows.push(["Memoria", (gpu.memory_used_mb / 1024).toFixed(1) + " di " + Math.round(gpu.memory_total_mb / 1024) + " GB"]);
      }
      if (gpu.temperature !== null) rows.push(["Temperatura", gpu.temperature + " °C"]);
      if (gpu.power_w !== null) rows.push(["Potenza", Math.round(gpu.power_w) + " W"]);
      rows.push(["Lavora a", busy ? busy + (busy === 1 ? " processo" : " processi") : "niente (libera)"]);
      statRows("gpu-sub", rows);
    } else {
      byId("gpu-value").textContent = "n/d";
      setBar("gpu-bar", 0);
      statRows("gpu-sub", [["Scheda", "nessuna GPU NVIDIA trovata"]]);
    }
    if (system.cpus) {
      const cpu = system.cpu;
      if (cpu) {
        byId("cpu-value").textContent = Math.round(cpu.total) + "%";
        pushUsage("cpu", cpu.total);
        renderCores(cpu.cores);
      } else {
        byId("cpu-value").textContent = "…";
      }
      renderSpark("cpu-spark", usage.cpu, "var(--spark-cpu)");
      const rows = cpu ? [["Core occupati", cpu.cores.filter((value) => value >= 50).length + " di " + system.cpus]] : [["Core", String(system.cpus)]];
      if (system.loads) {
        const [one, , fifteen] = system.loads;
        const perCore = fifteen / system.cpus;
        rows.push(["Ultimi 15 minuti", perCore < 0.25 ? "leggero" : perCore < 0.7 ? "moderato" : "pesante"]);
        rows.push(["Andamento", one > fifteen * 1.3 + 0.5 ? "in salita" : one < fifteen * 0.7 - 0.5 ? "in calo" : "stabile"]);
      }
      statRows("cpu-sub", rows);
    }
    const memory = system.memory;
    if (memory) {
      byId("ram-value").textContent = memory.available_gb + " GB liberi";
      setBar("ram-bar", memory.used_percent, memory.used_percent > 90 ? "bad" : memory.used_percent > 75 ? "warn" : "");
      statRows("ram-sub", [
        ["In uso", (memory.total_gb - memory.available_gb).toFixed(1) + " GB (" + memory.used_percent + "%)"],
        ["Totale", memory.total_gb + " GB"]
      ]);
    } else {
      byId("ram-value").textContent = "n/d";
    }
    const disk = system.disk;
    if (disk) {
      const used = 100 - disk.free_gb * 100 / disk.total_gb;
      byId("disk-value").textContent = disk.free_gb + " GB liberi";
      setBar("disk-bar", used, disk.free_gb < 10 ? "bad" : used > 80 ? "warn" : "");
      statRows("disk-sub", [
        ["In uso", (disk.total_gb - disk.free_gb).toFixed(1) + " GB (" + Math.round(used) + "%)"],
        ["Totale", disk.total_gb + " GB"]
      ]);
    }
    const processes = gpu ? gpu.process_list : [];
    if (unchanged("gpu-procs", processes)) return;
    byId("gpu-procs-empty").hidden = processes.length > 0;
    byId("gpu-procs").replaceChildren(...processes.map((process) => {
      const row = node("tr");
      row.append(
        node("td", "mono", process.pid === null ? "?" : String(process.pid)),
        node("td", "", process.name),
        node("td", "num", process.memory_mb === null ? "–" : Math.round(process.memory_mb) + " MiB")
      );
      return row;
    }));
  }
  function jobItem(job) {
    const item = node("li", "item");
    const [label, kind] = JOB_STATE[job.state];
    const text = node("div", "grow");
    text.append(node("div", "what", job.label), node("div", "sub", job.started));
    const view = node("button", "btn small", "Dettagli");
    view.type = "button";
    view.addEventListener("click", () => openLog(job.id, job.label));
    item.append(pill(label, kind), text, view);
    if (job.stoppable) {
      const stop = node("button", "btn small danger", "Ferma");
      stop.type = "button";
      stop.addEventListener("click", () => void stopJob(job));
      item.appendChild(stop);
    }
    return item;
  }
  function renderRunning(current) {
    const prep = current.preparation;
    const running = current.jobs.filter((job) => job.state === "running");
    if (unchanged("running", [running, current.activities, prep])) return;
    const fromSite = new Set(running.map((job) => job.label));
    const items = running.map(jobItem);
    for (const activity of current.activities) {
      if (fromSite.has(activity.label)) continue;
      const row = node("li", "item");
      row.append(pill("in corso", "run"), node("span", "what grow", activity.label), node("span", "sub", "pid " + activity.pid));
      if (activity.label === "preparazione" && prep && prep.percent !== null) {
        const bar = node("div", "bar run");
        const fill2 = node("span");
        fill2.style.width = prep.percent + "%";
        bar.appendChild(fill2);
        row.append(bar, node(
          "span",
          "sub",
          prep.routes_done + " di " + prep.routes_total + " route pronte · " + prep.percent + "%"
        ));
      }
      items.push(row);
    }
    byId("running").replaceChildren(...items);
    byId("running-empty").hidden = items.length > 0;
  }
  function renderJobs(current) {
    if (unchanged("jobs", current.jobs)) return;
    byId("jobs-empty").hidden = current.jobs.length > 0;
    byId("jobs").replaceChildren(...current.jobs.map(jobItem));
  }
  function renderUpdate(current) {
    const state = current.actions.update;
    if (unchanged("update", state)) return;
    const wrap = node("div", "action");
    const button = node("button", "btn", state.label);
    button.type = "button";
    button.disabled = !state.allowed;
    button.addEventListener("click", () => void run("update"));
    wrap.appendChild(button);
    if (!state.allowed && state.reason) {
      const reason = node("div", "reason", state.reason);
      reason.id = "reason-update";
      button.setAttribute("aria-describedby", reason.id);
      wrap.appendChild(reason);
    }
    byId("update-buttons").replaceChildren(wrap);
  }
  function render(current) {
    byId("readonly").hidden = current.actions_enabled;
    renderMachine(current);
    renderRunning(current);
    renderJobs(current);
    renderUpdate(current);
    if (log.id) {
      const job = current.jobs.find((item) => item.id === log.id);
      byId("sheet-state").hidden = !(job && job.state === "running");
    }
  }
  function showLive(text, ok) {
    byId("live-dot").className = ok ? "dot" : "dot off";
    byId("live-dot").title = text;
    byId("live-text").textContent = text;
  }
  async function refresh() {
    try {
      status = await api("/api/panel/status");
      render(status);
      showLive("aggiornato " + clock(), true);
    } catch (error) {
      showLive("non raggiungibile: " + messageOf(error), false);
    }
  }
  async function run(name) {
    if (!status) return;
    if (!window.confirm(status.actions[name].label + ": git pull e riavvio del sito (qualche secondo). Avvio?")) return;
    try {
      const job = await post("/api/panel/run/" + name);
      toast("Partito: " + job.label);
      await refresh();
      openLog(job.id, job.label);
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }
  async function stopJob(job) {
    if (!window.confirm('Fermo "' + job.label + '"? Il lavoro già fatto resta, si riprende rilanciando.')) return;
    try {
      await post("/api/panel/job/" + job.id + "/stop");
      toast("Fermato: " + job.label);
    } catch (error) {
      toast(messageOf(error), true);
    }
    void refresh();
  }
  function openLog(id, label) {
    window.clearTimeout(log.timer);
    log = { id, offset: 0, timer: void 0, follow: true };
    byId("sheet-title").textContent = label;
    byId("sheet-text").textContent = "";
    byId("sheet-follow").setAttribute("aria-pressed", "true");
    byId("sheet").classList.add("open");
    byId("sheet-backdrop").classList.add("open");
    document.body.classList.add("sheet-open");
    void pollLog();
  }
  function closeLog() {
    window.clearTimeout(log.timer);
    log.id = null;
    byId("sheet").classList.remove("open");
    byId("sheet-backdrop").classList.remove("open");
    document.body.classList.remove("sheet-open");
  }
  async function pollLog() {
    const id = log.id;
    if (!id) return;
    try {
      const data = await api("/api/panel/job/" + id + "/log?offset=" + log.offset);
      if (log.id !== id) return;
      const pre = byId("sheet-text");
      if (data.text) {
        if (log.offset === 0 && data.offset - data.text.length > 0) pre.textContent = "… (inizio tagliato)\n";
        pre.textContent += data.text;
        if (log.follow) pre.scrollTop = pre.scrollHeight;
      }
      log.offset = data.offset;
    } catch (error) {
      toast(messageOf(error), true);
      return;
    }
    const job = status?.jobs.find((item) => item.id === id);
    const running = !job || job.state === "running";
    byId("sheet-state").hidden = !running;
    log.timer = window.setTimeout(() => void pollLog(), running ? 1500 : 5e3);
  }
  byId("sheet-close").addEventListener("click", closeLog);
  byId("sheet-backdrop").addEventListener("click", closeLog);
  byId("sheet-follow").addEventListener("click", (event) => {
    log.follow = !log.follow;
    event.currentTarget.setAttribute("aria-pressed", String(log.follow));
    if (log.follow) byId("sheet-text").scrollTop = byId("sheet-text").scrollHeight;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && log.id) closeLog();
  });
  initAccountMenu();
  void refresh();
  window.setInterval(() => void refresh(), 4e3);
})();
