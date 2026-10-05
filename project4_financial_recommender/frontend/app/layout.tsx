import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Financial Product Recommender",
  description: "Eligibility-aware, explainable financial product recommendations (learning project)",
};

/** Root HTML shell for every page; wraps content in the TanStack Query provider. */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
