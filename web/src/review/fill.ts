// v-fill: the value of a field of the panel. A field the reviewer is typing in keeps what is typed: only the
// others follow the data (a plain :value would put the data back at every drawing).
import type {Directive} from 'vue';

type Field = HTMLInputElement | HTMLSelectElement;

export const vFill: Directive<Field, string> = {
  mounted(element, binding) {
    element.value = binding.value;
  },
  updated(element, binding) {
    if (document.activeElement !== element) element.value = binding.value;
  },
};
