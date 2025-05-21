import { GeistSans, GeistMono } from "geist/font";
import "./globals.css";

const geistSans = GeistSans;
const geistMono = GeistMono;

export const metadata = {
  title: "Adaptive RAG Learning System",
  description: "A frontend for an adaptive learning system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      {/* Ensure no whitespace between <html> and <body>, and <body> and {children} */}
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>{children}</body>
    </html>
  );
}
