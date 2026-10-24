// Single-file components for the type checker (vue-tsc reads the real types of each .vue file).
declare module '*.vue' {
  import type {DefineComponent} from 'vue';
  const component: DefineComponent<object, object, unknown>;
  export default component;
}
