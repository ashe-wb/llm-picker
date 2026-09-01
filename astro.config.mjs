// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://llm-picker.dev',
  vite: {
    plugins: [tailwindcss()],
  },
});
