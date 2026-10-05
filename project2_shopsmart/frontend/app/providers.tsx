"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/** Client-side provider that gives every component access to one shared TanStack Query cache. */
export function Providers({ children }: { children: React.ReactNode }) {
  // useState's initialiser runs once, so the QueryClient (and its cache) survives re-renders.
  // Data counts as fresh for 30s, and switching browser tabs doesn't trigger refetches.
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
