// Revisione cartelli (review/ReviewApp.vue), with the frames embedded in the page: the payload of
// pages.render_review in <script id="payload">. The page draws its own top bar, the Account menu in it.
import {mountPage} from '../common/mount';
import ReviewApp from '../review/ReviewApp.vue';
import type {Payload} from '../review/types';

const payload = JSON.parse(document.getElementById('payload')?.textContent || '{}') as Partial<Payload>;
mountPage(ReviewApp, {payload});
