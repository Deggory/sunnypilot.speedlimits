"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
  var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

  // web/src/common/api.ts
  var ApiError = class extends Error {
    constructor(message, status) {
      super(message);
      __publicField(this, "status", status);
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
  function thousands(value) {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  }
  function when(started) {
    const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})/.exec(started);
    if (!match) return started;
    const [, year, month, day, hour, minute] = match;
    const now = /* @__PURE__ */ new Date();
    const today = Number(year) === now.getFullYear() && Number(month) === now.getMonth() + 1 && Number(day) === now.getDate();
    return (today ? "" : day + "/" + month + " ") + hour + ":" + minute;
  }
  function clock() {
    return (/* @__PURE__ */ new Date()).toLocaleTimeString("it-IT");
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

  // web/src/panel/actions.ts
  function confirmed(name, label) {
    if (name === "train" || name === "train_continue") {
      return window.confirm(label + ": il training usa la GPU per ore. Avvio?");
    }
    if (name === "fetch") return window.confirm(label + ": scarica dal comma i giri nuovi. Avvio?");
    return true;
  }
  async function run(name, state, context2) {
    if (!confirmed(name, state.label)) return;
    try {
      const job = await post("/api/panel/run/" + name);
      context2.started(job);
      toast("Partito: " + job.label);
      await context2.refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void context2.refresh();
    }
  }
  async function stopJob(job, context2) {
    if (!window.confirm('Fermo "' + job.label + '"? Il lavoro già fatto resta, si riprende rilanciando.')) return;
    try {
      await post("/api/panel/job/" + job.id + "/stop");
      toast("Fermata: " + job.label);
    } catch (error) {
      toast(messageOf(error), true);
    }
    void context2.refresh();
  }
  function actionButton(name, state, primary, context2) {
    const button = node("button", "btn" + (primary ? " primary" : ""), state.label);
    button.type = "button";
    button.disabled = !state.allowed;
    button.addEventListener("click", () => void run(name, state, context2));
    return button;
  }
  function fillActions(buttonsId, reasonId, names, actions, context2) {
    const blocked = names.filter((name) => !actions[name].allowed && actions[name].reason);
    const reasons = [...new Set(blocked.map((name) => actions[name].reason))];
    const only = reasons.length === 1 ? reasons[0] : void 0;
    const text = only !== void 0 ? only.charAt(0).toUpperCase() + only.slice(1) + "." : blocked.map((name) => actions[name].label + ": " + actions[name].reason).join(" · ");
    const reason = byId(reasonId);
    reason.textContent = text;
    reason.title = text;
    byId(buttonsId).replaceChildren(...names.map((name) => {
      const button = actionButton(name, actions[name], false, context2);
      if (blocked.includes(name)) button.setAttribute("aria-describedby", reasonId);
      return button;
    }));
  }

  // web/src/panel/hero.ts
  var HERO = {
    prepare: ["Prepara le route nuove", "Ci sono video nuovi da trasformare in fotogrammi con i cartelli proposti."],
    review: ["Revisiona i cartelli", "Apri le route da revisionare e conferma o correggi i cartelli."],
    train: ["Allena il modello", "Le revisioni sono pronte: importale e allena detector e reader sulla GPU."],
    fetch: ["Scarica nuovi giri", "Per allenare servono più route: scarica altri giri dal comma."]
  };
  var ATTENTION = {
    prepare: "step-prepare",
    review: "step-review",
    train: "step-train"
  };
  var STEP_CARDS = ["step-prepare", "step-review", "step-train"];
  function renderHero(status, context2) {
    const action = status.next_action;
    let kind = "done";
    let [title, text] = ["Tutto in ordine", status.next_step];
    if (action) {
      [title, text] = HERO[action];
      kind = action === "fetch" ? "prepare" : action;
    }
    if (action === "train" && status.next_step.startsWith("tutto pronto")) {
      title = "Tutto pronto";
      text = "Il dataset è aggiornato: puoi allenare un nuovo modello quando vuoi.";
      kind = "done";
    }
    byId("hero").className = "card hero k-" + kind;
    byId("hero-title").textContent = title;
    byId("hero-text").textContent = text;
    const actions = byId("hero-actions");
    actions.replaceChildren();
    if (action === "review") {
      const link = node("a", "btn primary", "Apri la revisione →");
      link.href = "/";
      actions.appendChild(link);
    } else if (action) {
      const state = status.actions[action];
      const button = actionButton(action, state, true, context2);
      actions.appendChild(button);
      if (!state.allowed && state.reason) {
        const reason = node("p", "reason", state.reason);
        reason.id = "hero-reason";
        button.setAttribute("aria-describedby", reason.id);
        actions.appendChild(reason);
      }
    }
    const attention = action ? ATTENTION[action] : void 0;
    for (const id of STEP_CARDS) byId(id).classList.toggle("attention", id === attention);
  }

  // web/src/panel/jobs.ts
  var BOXES = [
    ["prepare-live", ["fetch", "prepare"]],
    ["train-live", ["dataset", "train", "train_continue"]]
  ];
  var JOB_STATE = {
    running: ["in corso", "run"],
    ok: ["finito", "ok"],
    failed: ["fallito", "bad"],
    interrupted: ["interrotto", "warn"]
  };
  var RECENT_MINUTES = 60;
  var TAIL_BYTES = 6e3;
  var CLOSED_KEY = "nnslr-panel-closed";
  var cards = {
    /** Jobs seen running here (or started here) stay after they end, until closed. */
    seen: /* @__PURE__ */ new Set(),
    /** Closed by the user, for this browser tab. */
    dismissed: loadDismissed(),
    tails: /* @__PURE__ */ new Map(),
    shown: /* @__PURE__ */ new Map(),
    /** The last status and context, to draw again when a card is closed. */
    last: null
  };
  function loadDismissed() {
    try {
      const stored = JSON.parse(sessionStorage.getItem(CLOSED_KEY) ?? "[]");
      return new Set(Array.isArray(stored) ? stored.filter((id) => typeof id === "string") : []);
    } catch {
      return /* @__PURE__ */ new Set();
    }
  }
  function dismiss(id) {
    cards.dismissed.add(id);
    try {
      sessionStorage.setItem(CLOSED_KEY, JSON.stringify([...cards.dismissed]));
    } catch {
    }
    if (cards.last) renderJobCards(cards.last.status, cards.last.context);
  }
  function followJob(id) {
    cards.seen.add(id);
  }
  function startedAt(job) {
    const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})/.exec(job.started);
    if (!match) return null;
    const [year = 0, month = 1, day = 1, hour = 0, minute = 0, second = 0] = match.slice(1).map(Number);
    return new Date(year, month - 1, day, hour, minute, second);
  }
  function duration(ms) {
    const minutes = Math.floor(ms / 6e4);
    if (minutes < 1) return "meno di un minuto";
    if (minutes < 60) return minutes + " min";
    return Math.floor(minutes / 60) + " h " + minutes % 60 + " min";
  }
  function liveJob(status, actions) {
    const jobs = status.jobs.filter((job) => actions.includes(job.action) && !cards.dismissed.has(job.id));
    const running = jobs.find((job) => job.state === "running");
    if (running) {
      cards.seen.add(running.id);
      return running;
    }
    const last = jobs[0];
    if (!last) return null;
    const start = startedAt(last);
    const recent = start !== null && Date.now() - start.getTime() < RECENT_MINUTES * 6e4;
    return cards.seen.has(last.id) || recent ? last : null;
  }
  async function readTail(job) {
    const known = cards.tails.get(job.id);
    if (known && known.size === job.size) return known;
    const from = Math.max(0, job.size - TAIL_BYTES);
    const data = await api("/api/panel/job/" + job.id + "/log?offset=" + from);
    let lines = data.text.split("\n");
    if (from > 0) lines = lines.slice(1);
    const useful = lines.map((line) => line.trim()).filter((line) => line && !line.startsWith("=="));
    const tail = {
      size: job.size,
      text: (from > 0 ? "… (inizio tagliato)\n" : "") + lines.join("\n"),
      line: useful[useful.length - 1] ?? ""
    };
    cards.tails.set(job.id, tail);
    return tail;
  }
  function buildCard(job, key, context2) {
    const [label, kind] = JOB_STATE[job.state];
    const root2 = node("div", "jobcard " + kind);
    const head = node("div", "jobcard-head");
    if (job.state === "running") head.appendChild(node("span", "spinner"));
    head.append(node("strong", "jobcard-label", job.label), pill(label, kind));
    const bar = node("div", "jobcard-bar");
    const fill = node("span");
    bar.appendChild(fill);
    const log = node("pre", "jobcard-log");
    const details = node("details");
    details.append(node("summary", "", "Dettagli tecnici"), log);
    const link = node("a", "", "Dettagli in Stato →");
    link.href = "/stato";
    const foot = node("div", "jobcard-foot");
    foot.append(link, node("span", "grow"));
    if (job.state === "running" && job.stoppable) {
      const stop = node("button", "btn small danger", "Ferma");
      stop.type = "button";
      stop.addEventListener("click", () => void stopJob(job, context2));
      foot.appendChild(stop);
    } else if (job.state !== "running") {
      const close = node("button", "btn small", "Chiudi");
      close.type = "button";
      close.addEventListener("click", () => dismiss(job.id));
      foot.appendChild(close);
    }
    foot.appendChild(details);
    const sub = node("div", "jobcard-sub");
    const line = node("p", "jobcard-line");
    root2.append(head, bar, sub, line, foot);
    return { key, root: root2, sub, bar, fill, line, log, details };
  }
  function summary(job, status, percent) {
    if (job.state === "running") {
      const start = startedAt(job);
      let text = "In corso da " + (start ? duration(Date.now() - start.getTime()) : "–");
      const prep = status.preparation;
      if (percent !== null && prep) text += " · " + percent + "% · " + prep.routes_done + " di " + prep.routes_total + " route pronte";
      return text;
    }
    const at = "Partito alle " + when(job.started);
    if (job.state === "ok") return at + ", finito.";
    if (job.state === "failed") return at + ", fallito" + (job.code !== null ? " (codice " + job.code + ")." : ".");
    return at + ", fermato.";
  }
  function showTail(card, tail) {
    card.line.textContent = tail ? tail.line : "";
    card.log.textContent = tail ? tail.text : "";
  }
  function renderBox(boxId, actions, status, context2) {
    const box = byId(boxId);
    const job = liveJob(status, actions);
    if (!job) {
      box.replaceChildren();
      cards.shown.delete(boxId);
      return;
    }
    const key = job.id + ":" + job.state + ":" + job.stoppable;
    let card = cards.shown.get(boxId);
    if (!card || card.key !== key) {
      const open = card?.details.open ?? false;
      card = buildCard(job, key, context2);
      card.details.open = open;
      cards.shown.set(boxId, card);
      box.replaceChildren(card.root);
    }
    const progress = job.action === "prepare" ? status.preparation?.percent ?? null : null;
    card.sub.textContent = summary(job, status, progress);
    card.bar.classList.toggle("indeterminate", job.state === "running" && progress === null);
    card.fill.style.width = (job.state === "ok" ? 100 : progress ?? 0) + "%";
    showTail(card, cards.tails.get(job.id));
    const shown2 = card;
    readTail(job).then((tail) => {
      if (cards.shown.get(boxId) === shown2) showTail(shown2, tail);
    }).catch(() => {
    });
  }
  function renderJobCards(status, context2) {
    cards.last = { status, context: context2 };
    for (const [boxId, actions] of BOXES) renderBox(boxId, actions, status, context2);
  }

  // web/src/panel/routes.ts
  var STATE = {
    collegare: ["da collegare", "todo"],
    estrarre: ["da estrarre", "todo"],
    preannotare: ["da pre-annotare", "todo"],
    revisionare: ["da revisionare", "muted hollow"],
    importare: ["da importare", "ok hollow", "pronta "],
    pronta: ["nel dataset", "ok"]
  };
  function statePill(label, kind, prefix) {
    const element = node("span", "pill " + kind);
    if (prefix) element.appendChild(node("span", "pill-long", prefix));
    element.append(label);
    return element;
  }
  function routeLink(route) {
    const link = node("a");
    link.href = "/route/" + encodeURIComponent(route) + "/";
    link.title = route;
    const pad = /^0+(?=[0-9a-f]+--)/.exec(route)?.[0];
    if (pad) link.append(node("span", "rid-pad", pad), route.slice(pad.length));
    else link.textContent = route;
    return link;
  }
  function reviewedCell(row) {
    if (!row.run_id) return node("td", "none", "–");
    const bar = node("div", "bar ok");
    const fill = node("span");
    fill.style.width = (row.report_frames ? Math.min(100, row.confirmed * 100 / row.report_frames) : 0) + "%";
    bar.appendChild(fill);
    const box = node("div", "reviewed");
    box.append(bar, node("span", "num", thousands(row.confirmed) + " / " + thousands(row.report_frames)));
    const cell = node("td");
    cell.appendChild(box);
    return cell;
  }
  function stateCell(row) {
    const [label, kind, prefix] = STATE[row.state];
    const cell = node("td");
    cell.appendChild(statePill(label, kind, prefix));
    if (row.state === "importare") cell.appendChild(node("span", "note", thousands(row.new_reviews) + " fotogrammi nuovi"));
    if (row.run_id && row.segments > row.extracted) {
      const late = row.segments - row.extracted;
      cell.appendChild(node("span", "note", late + (late === 1 ? " segmento arrivato" : " segmenti arrivati") + " dopo, non usati"));
    }
    return cell;
  }
  function routeRow(row) {
    const id = node("td", "route");
    id.appendChild(routeLink(row.route));
    const tr = node("tr");
    tr.append(
      id,
      node("td", "num", row.extracted + " / " + row.segments),
      node("td", "num", thousands(row.frames)),
      reviewedCell(row),
      node("td", "num", thousands(row.objects)),
      stateCell(row)
    );
    return tr;
  }
  function renderRoutes(routes) {
    byId("routes-empty").hidden = routes.length > 0;
    byId("routes-wrap").hidden = routes.length === 0;
    byId("routes").replaceChildren(...routes.map(routeRow));
  }

  // web/src/panel/steps.ts
  var TO_PREPARE = ["collegare", "estrarre", "preannotare"];
  var REVIEW_SUB = "Confermi o correggi i cartelli: l'unico passo a mano.";
  var TRAIN_SUB = "Revisioni → dataset → modello, sulla GPU.";
  var sum = (rows, value) => rows.reduce((total, row) => total + value(row), 0);
  function big(id, figure, note) {
    byId(id).replaceChildren(figure, node("span", "big-note", note));
  }
  function renderPrepare(status, context2) {
    const toPrepare = status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
    if (toPrepare) big("prepare-big", toPrepare + (toPrepare === 1 ? " route nuova" : " route nuove"), "da trasformare in fotogrammi");
    else big("prepare-big", "Tutto preparato", "nessun video in attesa");
    fillActions("prepare-buttons", "prepare-reason", ["fetch", "prepare"], status.actions, context2);
  }
  function renderReview(status) {
    const reported = status.routes.filter((row) => row.run_id);
    const frames = sum(reported, (row) => row.report_frames);
    const confirmed2 = sum(reported, (row) => Math.min(row.confirmed, row.report_frames));
    const toReview = status.routes.filter((row) => row.state === "revisionare").length;
    if (frames) big("review-big", thousands(confirmed2) + " di " + thousands(frames), "fotogrammi confermati");
    else byId("review-big").textContent = "Niente da revisionare";
    setBar("review-bar", frames ? confirmed2 * 100 / frames : 0, "ok");
    byId("review-sub").textContent = !frames ? REVIEW_SUB : toReview ? toReview + " route ancora da revisionare." : "Tutte le route sono revisionate.";
  }
  function renderTrain(status, context2) {
    const reviewed = status.reviewed_routes;
    const needed = status.min_routes;
    byId("train-goal").replaceChildren(
      ...Array.from({ length: needed }, (_, index) => node("span", index < reviewed ? "on" : ""))
    );
    if (reviewed >= needed) big("train-big", reviewed + " route revisionate", "ne bastano " + needed + " per allenare");
    else big("train-big", reviewed + " di " + needed + " route", "revisionate, ne servono " + needed + " per allenare");
    const fresh = sum(status.routes, (row) => row.new_reviews);
    byId("train-sub").textContent = (fresh ? thousands(fresh) + " fotogrammi confermati da importare. " : "") + (reviewed < needed ? "Lo split train/validation/test ne vuole almeno " + needed + " indipendenti." : TRAIN_SUB);
    fillActions("train-buttons", "train-reason", ["dataset", "train", "train_continue"], status.actions, context2);
  }
  function renderSteps(status, context2) {
    renderPrepare(status, context2);
    renderReview(status);
    renderTrain(status, context2);
  }

  // web/src/pages/panel.ts
  var context = { refresh, started: (job) => followJob(job.id) };
  function render(status) {
    byId("readonly").classList.toggle("show", !status.actions_enabled);
    if (!unchanged("panel", [
      status.routes,
      status.actions,
      status.jobs,
      status.next_step,
      status.next_action,
      status.reviewed_routes,
      status.min_routes,
      status.actions_enabled
    ])) {
      renderHero(status, context);
      renderSteps(status, context);
      renderRoutes(status.routes);
    }
    renderJobCards(status, context);
  }
  function showLive(text, ok) {
    document.querySelector(".live")?.classList.toggle("off", !ok);
    const dot = byId("live-dot");
    dot.className = ok ? "dot" : "dot off";
    dot.title = text;
    byId("live-text").textContent = text;
  }
  async function refresh() {
    try {
      render(await api("/api/panel/status"));
      showLive("aggiornato " + clock(), true);
    } catch (error) {
      showLive("non raggiungibile: " + messageOf(error), false);
    }
  }
  initThemeSwitch();
  void refresh();
  window.setInterval(() => void refresh(), 4e3);
})();
