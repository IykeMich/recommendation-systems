import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

/** Page <title> and meta description (Next.js App Router metadata). */
export const metadata: Metadata = {
  title: "RecLab",
  description: "Recommendation playground for the RecLab lab pack",
};

/** Root HTML shell shared by every page; wraps content in the React Query provider. */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
