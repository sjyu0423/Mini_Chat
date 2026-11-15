export interface UiSettings {
  temperature: number;
  maxNewTokens: number;
}

/** M5 recommended configuration (top-k 40 and repetition penalty 1.2 are fixed server-side). */
export const DEFAULT_SETTINGS: UiSettings = { temperature: 0.7, maxNewTokens: 200 };
