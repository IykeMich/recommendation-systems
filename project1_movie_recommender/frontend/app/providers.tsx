"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/** Client-side wrapper that gives the whole app a shared TanStack Query cache. */
export function Providers({ children }: { children: React.ReactNode }) {
  // useState initialiser creates the QueryClient once per browser session (not on every render).
  // Cached data counts as fresh for 60s, and switching browser tabs doesn't trigger refetches.
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 60_000, refetchOnWindowFocus: false },
        },
      }),
  );
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
