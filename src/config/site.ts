/**
 * Single source of branding truth. Renaming the site later = edit this file
 * (+ the `site` field in astro.config.mjs) and nothing else.
 */
export const SITE = {
  name: 'llm-picker',
  tagline: 'What LLM your machine can run — and which one is a good fit for a specific job.',
  description:
    'Find which local LLMs fit your machine, based on memory, quantization band and context length, then which of them suit your workload. Each model lists its known issues and cited workarounds.',
  domain: 'https://llm-picker.dev',
  repoUrl: 'https://github.com/ashe-wb/llm-picker',
  updateCadenceDays: 7,
} as const;
