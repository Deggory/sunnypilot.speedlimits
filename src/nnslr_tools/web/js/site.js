"use strict";
(() => {
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

  // web/src/pages/site.ts
  initThemeSwitch();
})();
