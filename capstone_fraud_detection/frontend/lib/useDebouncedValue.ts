import { useEffect, useState } from "react";

/** Returns `value` once it has stopped changing for `delayMs`. */
export function useDebouncedValue<Value>(value: Value, delayMs = 250): Value {
  const [debouncedValue, setDebouncedValue] = useState(value);

  // Every change restarts the timer; the cleanup cancels the pending update, so only the
  // last value in a burst (e.g. while a slider is being dragged) is ever published.
  useEffect(() => {
    const timeoutId = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timeoutId);
  }, [value, delayMs]);

  return debouncedValue;
}
