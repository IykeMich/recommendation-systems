import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

// Next.js reads this export to set the page <title> and meta description.
export const metadata: Metadata = {
  title: "Fraud Ops",
  description: "Fraud-risk operations dashboard for the capstone (synthetic data)",
};

/**
 * Root layout (a server component) wrapping every page. It loads global CSS and puts the
 * client-side Providers (TanStack Query) around the page content.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
