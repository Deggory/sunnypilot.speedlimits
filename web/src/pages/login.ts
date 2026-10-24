// The sign-in page (login/LoginApp.vue); signed out, there is no Account menu.
import {createApp} from 'vue';
import LoginApp from '../login/LoginApp.vue';

const host = document.getElementById('app');
if (host !== null) createApp(LoginApp).mount(host);
