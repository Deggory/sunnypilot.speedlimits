// Builds every page of web/src/pages/<page>.ts into one self-contained script, <out>/<page>.js
// (an IIFE: pages.py puts it inline with the page's nonce). Vue's single-file components are compiled
// here, so the browser never compiles templates (the CSP allows no eval).
// Usage: node build.mjs <out> [page ...]   (no page: all of them)
import {readdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {build} from 'vite';
import vue from '@vitejs/plugin-vue';

const out = resolve(process.argv[2] ?? '../src/nnslr_tools/web/js');
const wanted = process.argv.slice(3);
const pages = readdirSync('src/pages').filter((name) => name.endsWith('.ts')).map((name) => name.slice(0, -3))
  .filter((page) => wanted.length === 0 || wanted.includes(page)).sort();

for (const page of pages) {
  await build({
    configFile: false,
    logLevel: 'warn',
    plugins: [vue()],
    define: {
      'process.env.NODE_ENV': '"production"',
      __VUE_OPTIONS_API__: 'false',
      __VUE_PROD_DEVTOOLS__: 'false',
      __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false',
    },
    build: {
      outDir: out,
      emptyOutDir: false,
      target: 'es2020',
      minify: true,
      reportCompressedSize: false,
      // The styles stay in the page templates and shared.css: a component must not emit CSS.
      cssCodeSplit: false,
      lib: {entry: resolve('src/pages', page + '.ts'), formats: ['iife'], name: 'nnslr_' + page, fileName: () => page + '.js'},
    },
  });
}
console.log('Script delle pagine: ' + pages.join(', '));
