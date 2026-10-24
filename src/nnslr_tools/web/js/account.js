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

  // web/src/pages/account.ts
  var choice = byId("theme-choice");
  function showTheme() {
    const theme = storedTheme();
    for (const button of choice.querySelectorAll("[data-theme-choice]")) {
      button.setAttribute("aria-checked", String(button.dataset["themeChoice"] === theme));
    }
  }
  choice.addEventListener("click", (event) => {
    const button = event.target.closest("[data-theme-choice]");
    const theme = button?.dataset["themeChoice"];
    if (theme === "dark" || theme === "light" || theme === "auto") setTheme(theme);
  });
  onThemeChange(showTheme);
  showTheme();
  function render(state) {
    let text = "Connesso come " + (state.user ?? "") + ".";
    if (state.sessions !== void 0 && state.sessions > 1) text += " Connesso da " + state.sessions + " browser.";
    byId("admin-sub").textContent = text;
  }
  async function signOut(everywhere) {
    if (everywhere && !window.confirm("Uscire da tutti i browser, compreso questo?")) return;
    try {
      await post(everywhere ? "/api/auth/logout-all" : "/api/auth/logout");
    } catch {
    }
    location.replace("/");
  }
  byId("btn-signout").addEventListener("click", () => void signOut(false));
  byId("btn-signout-all").addEventListener("click", () => void signOut(true));
  byId("password-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const error = byId("password-error");
    const fields = ["current", "fresh", "repeat"].map((id) => byId(id));
    const [current, fresh, repeat] = fields.map((field) => field.value);
    const button = byId("btn-password");
    error.textContent = "";
    byId("password-ok").textContent = "";
    if (!current) {
      error.textContent = "Scrivi la password attuale.";
      return;
    }
    if (!fresh || fresh.length < 12) {
      error.textContent = "La nuova password deve avere almeno 12 caratteri.";
      return;
    }
    if (fresh !== repeat) {
      error.textContent = "Le due nuove password sono diverse.";
      return;
    }
    button.disabled = true;
    try {
      const answer = await post("/api/auth/password", { current, password: fresh });
      for (const field of fields) field.value = "";
      render(answer.state);
      byId("password-ok").textContent = "Password cambiata: gli altri browser sono usciti.";
    } catch (problem) {
      error.textContent = messageOf(problem);
    } finally {
      button.disabled = false;
    }
  });
  initThemeSwitch();
  api("/api/auth/state").then(render).catch((problem) => {
    byId("admin-sub").textContent = messageOf(problem);
  });
})();
