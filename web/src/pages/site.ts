// The pages built by pages.py without a script of their own (route list, error pages): the Account menu and
// the confirm dialog; on the route list also the "Review with AI" sheet of its buttons (button[data-ai-route]).
import {createApp} from 'vue';
import {mountSiteParts} from '../common/mount';
import AiRouteSheet from '../components/AiRouteSheet.vue';

mountSiteParts();
if (document.querySelector('button[data-ai-route]') !== null) {
  const host = document.createElement('div');
  host.id = 'ai-route-host';
  document.body.append(host);
  createApp(AiRouteSheet).mount(host);
}
