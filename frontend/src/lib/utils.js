import { clsx } from "clsx";
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Minimal LaTeX formatter - just fix the essential issues
 * @param {string} text - The text containing LaTeX commands
 * @returns {string} - Formatted text for KaTeX rendering
 */
export const cleanLatexString = (text) => {
  if (!text || typeof text !== 'string') {
    return text;
  }
  
  // Debug log to see what's happening
  console.log('Input text:', text);
  
  const result = text
    // Only fix double backslashes from JSON encoding
    .replace(/\\\\/g, '\\')
    // Clean up excessive whitespace but preserve structure
    .replace(/[ \t]+/g, ' ')
    // Don't collapse line breaks that might be important
    .replace(/\n\s*\n/g, '\n\n')
    .trim();
  
  console.log('Output text:', result);
  return result;
};

/**
 * Debug utility to test LaTeX processing
 * @param {string} input - Raw LaTeX string
 * @returns {object} - Object with original, processed, and debug info
 */
export const debugLatex = (input) => {
  const original = input;
  const processed = cleanLatexString(input);
  
  return {
    original,
    processed,
    changes: {
      hasBackslashes: /\\\\/g.test(original),
      hasMath: /\$/.test(original),
      hasDisplayMath: /\$\$/.test(processed),
      mathBlocks: processed.match(/\$[^$]*\$/g) || [],
      displayMathBlocks: processed.match(/\$\$[^$]*\$\$/g) || []
    }
  };
};

// Test cases for common edge cases
export const testLatexCases = () => {
  const testCases = [
    "$\\\\ast$",
    "Given that $\\\\ast$ is an operation",
    "\\\\frac{a}{b}",
    "$x^2 + y^2$",
    "\\\\[x = \\\\frac{-b \\\\pm \\\\sqrt{b^2-4ac}}{2a}\\\\]",
    "\\\\(a + b\\\\)",
    "$\\\\mathbb{R}$",
    "$\\\\operatorname{ker}(f)$"
  ];
  
  return testCases.map(testCase => debugLatex(testCase));
};
