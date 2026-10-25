// The newest "Try without saving" of the route (GET …/ai-trial, nnslr_tools.ai_review.latest_trial): the frames it
// has a suggestion for, for the "AI trial" filter. ai.ts reads it and shows each suggestion as an AI answer.
import {reactive} from 'vue';

/** The frames of the trial that nobody had confirmed when the page read it. */
export const trialKeys: Set<string> = reactive(new Set<string>());
