import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "ShopSmart Offers",
  description: "Context-aware offer recommendations for ShopSmart shoppers",
};

/** Root layout for every route: wraps pages in the TanStack Query provider. */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
