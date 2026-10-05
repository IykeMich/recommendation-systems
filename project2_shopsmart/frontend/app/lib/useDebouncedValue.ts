import { useEffect, useState } from "react";

/** Returns `value` once it has stopped changing for `delayMs`. */
export function useDebouncedValue<Value>(value: Value, delayMs = 250): Value {
  const [debouncedValue, setDebouncedValue] = useState(value);

  // Each change restarts the timer; the cleanup cancels the previous one, so only the last value lands.
  useEffect(() => {
    const timeoutId = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timeoutId);
  }, [value, delayMs]);

  return debouncedValue;
}
