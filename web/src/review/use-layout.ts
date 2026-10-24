// Sizes the CSS needs from the script, and the on-screen size of the pictures: the height of the top bar
// (the docked panel starts under it) and of the phone's bottom bar (the page ends above it); the size of the
// photo and of the zoom, which the labels and corner handles follow.
import {onBeforeUnmount, onMounted, ref, type Ref} from 'vue';

function observe(element: Readonly<Ref<Element | null>>, measure: (element: Element) => void): void {
  let observer: ResizeObserver | null = null;
  let onResize: (() => void) | null = null;
  onMounted(() => {
    const target = element.value;
    if (target === null) return;
    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(() => measure(target));
      observer.observe(target);
    } else {
      onResize = () => measure(target);
      window.addEventListener('resize', onResize);
      measure(target);
    }
  });
  onBeforeUnmount(() => {
    observer?.disconnect();
    if (onResize !== null) window.removeEventListener('resize', onResize);
  });
}

/** Keeps a CSS variable of the page at the height of this element (--topbar-h, --mobilebar-h). */
export function useHeightVar(element: Readonly<Ref<Element | null>>, property: string): void {
  observe(element, (target) => {
    document.documentElement.style.setProperty(property, Math.ceil(target.getBoundingClientRect().height) + 'px');
  });
}

/** The on-screen size of an element (an SVG picture), followed as it changes. */
export function useScreenSize(element: Readonly<Ref<Element | null>>): Ref<{width: number; height: number}> {
  const size = ref({width: 0, height: 0});
  observe(element, (target) => {
    const rect = target.getBoundingClientRect();
    if (rect.width !== size.value.width || rect.height !== size.value.height) {
      size.value = {width: rect.width, height: rect.height};
    }
  });
  return size;
}
