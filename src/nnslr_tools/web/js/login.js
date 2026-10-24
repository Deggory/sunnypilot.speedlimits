"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __defNormalProp = (obj, key, value2) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value: value2 }) : obj[key] = value2;
  var __publicField = (obj, key, value2) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value2);

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
    const show2 = () => {
      const dark = isDark();
      const title = dark ? "Passa al tema chiaro" : "Passa al tema scuro";
      button.classList.toggle("is-light", !dark);
      button.setAttribute("aria-pressed", String(!dark));
      button.title = title;
      button.setAttribute("aria-label", title);
    };
    button.addEventListener("click", () => setTheme(isDark() ? "light" : "dark"));
    onThemeChange(show2);
    show2();
  }

  // web/src/pages/login.ts
  var VIEWS = ["form-login", "form-code", "form-setup"];
  var setupCode = "";
  function show(view) {
    for (const id of VIEWS) byId(id).hidden = id !== view;
    const setup = view !== "form-login";
    byId("auth-steps").hidden = !setup;
    byId("auth-back").hidden = view !== "form-setup";
    byId("auth-head").classList.toggle("idle", !setup);
    const [code, password] = byId("auth-steps").children;
    if (code) code.className = view === "form-code" ? "on" : "";
    if (password) password.className = view === "form-setup" ? "on" : "";
    const title = byId("auth-title");
    const sub = byId("auth-sub");
    if (view === "form-login") {
      title.textContent = "Bentornato";
      sub.textContent = "Accedi per revisionare i cartelli.";
      byId("login-user").focus();
    } else if (view === "form-code") {
      title.textContent = "Configuriamo il sito";
      sub.textContent = "Prima il codice di configurazione di questo server.";
      byId("setup-code").focus();
    } else {
      title.textContent = "Crea l’amministratore";
      sub.textContent = "Il nome e la password con cui entrerai da ora in poi.";
      byId("setup-password").focus();
    }
  }
  function done() {
    location.reload();
  }
  var value = (id) => byId(id).value;
  function guard(form, button, error, action) {
    byId(form).addEventListener("submit", async (event) => {
      event.preventDefault();
      byId(error).textContent = "";
      byId(button).disabled = true;
      try {
        await action();
      } catch (problem) {
        byId(error).textContent = messageOf(problem);
      } finally {
        byId(button).disabled = false;
      }
    });
  }
  guard("form-login", "login-submit", "login-error", async () => {
    await post("/api/auth/login", { user: value("login-user"), password: value("login-password") }, true);
    done();
  });
  guard("form-code", "code-submit", "code-error", async () => {
    const code = value("setup-code").trim();
    if (!code) throw new Error("Scrivi il codice di configurazione.");
    await post("/api/auth/setup/check", { code }, true);
    setupCode = code;
    show("form-setup");
  });
  guard("form-setup", "setup-submit", "setup-error", async () => {
    const password = value("setup-password");
    if (password.length < 12) throw new Error("La password deve avere almeno 12 caratteri.");
    if (password !== value("setup-repeat")) throw new Error("Le due password sono diverse.");
    await post("/api/auth/setup", { code: setupCode, user: value("setup-user").trim(), password }, true);
    done();
  });
  byId("auth-back").addEventListener("click", () => show("form-code"));
  byId("auth-transport").hidden = location.protocol === "https:";
  initThemeSwitch();
  api("/api/auth/state", {}, true).then((state) => {
    if (state.signed_in) done();
    else show(state.configured ? "form-login" : "form-code");
  }).catch((problem) => {
    byId("auth-sub").textContent = messageOf(problem);
  });
})();
