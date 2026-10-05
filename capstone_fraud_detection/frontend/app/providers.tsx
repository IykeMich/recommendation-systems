"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * Creates the TanStack Query client (the shared cache for all API data) and provides it to the app.
 * It lives in useState so it is created once per browser session, not on every render.
 * Defaults: data counts as fresh for 30 s, and switching back to the tab doesn't trigger refetches
 * (the dashboard polls on its own timers instead).
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 30_000, refetchOnWindowFocus: false },
        },
      }),
  );
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
