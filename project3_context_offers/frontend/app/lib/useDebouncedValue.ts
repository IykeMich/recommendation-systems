import { useEffect, useState } from "react";

/** Returns `value` once it has stopped changing for `delayMs`. */
export function useDebouncedValue<Value>(value: Value, delayMs = 250): Value {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    // Each change cancels the pending timer, so only the last value in a burst is applied.
    const timeoutId = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timeoutId);
  }, [value, delayMs]);

  return debouncedValue;
}
