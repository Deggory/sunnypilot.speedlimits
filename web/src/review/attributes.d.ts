// The review page drives its state through data attributes (data-filter, data-key, data-index...), read by
// its handlers and by whoever drives the page with Playwright: allowed on every element in the templates.
export {};

declare module 'vue' {
  interface HTMLAttributes {
    [key: `data-${string}`]: string | number | boolean | undefined;
  }
  interface SVGAttributes {
    [key: `data-${string}`]: string | number | boolean | undefined;
  }
}
