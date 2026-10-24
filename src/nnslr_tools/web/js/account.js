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

  // web/src/pages/account.ts
  function render(state) {
    let text = "Connesso come " + (state.user ?? "") + ".";
    if (state.sessions !== void 0 && state.sessions > 1) text += " Connesso da " + state.sessions + " browser.";
    byId("admin-sub").textContent = text;
  }
  async function signOut2(everywhere) {
    if (everywhere && !window.confirm("Uscire da tutti i browser, compreso questo?")) return;
    try {
      await post(everywhere ? "/api/auth/logout-all" : "/api/auth/logout");
    } catch {
    }
    location.replace("/");
  }
  byId("btn-signout").addEventListener("click", () => void signOut2(false));
  byId("btn-signout-all").addEventListener("click", () => void signOut2(true));
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
  initAccountMenu();
  api("/api/auth/state").then(render).catch((problem) => {
    byId("admin-sub").textContent = messageOf(problem);
  });
})();
