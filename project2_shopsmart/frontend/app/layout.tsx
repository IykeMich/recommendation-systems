import type { Metadata } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "ShopSmart",
  description: "Collaborative product recommendations for ShopSmart shoppers",
};

/** Root layout for every page: wraps the app in the TanStack Query provider. */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
