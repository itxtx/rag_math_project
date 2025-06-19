import { clsx } from "clsx";
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Clean LaTeX string by removing LaTeX commands and formatting
 * @param {string} text - The text containing LaTeX commands
 * @returns {string} - Cleaned text without LaTeX formatting
 */
export const cleanLatexString = (text) => {
  if (!text || typeof text !== 'string') {
    return text;
  }
  
  return text
    .replace(/\\[a-zA-Z]+(\{[^}]*\})?/g, '') // Remove LaTeX commands with optional braces
    .replace(/\$[^$]*\$/g, '') // Remove inline math
    .replace(/\\\[[^\]]*\\\]/g, '') // Remove display math
    .replace(/\\\([^)]*\\\)/g, '') // Remove inline math with \( \)
    .replace(/\{|\}/g, '') // Remove remaining braces
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim();
};
