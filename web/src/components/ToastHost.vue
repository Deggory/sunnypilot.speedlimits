<script setup lang="ts">
import {toastState} from '../composables/toast';
</script>

<template>
  <div class="toast" :class="{show: toastState !== null, error: toastState?.error}" id="toast" role="alert">
    {{ toastState?.message }}
  </div>
</template>
