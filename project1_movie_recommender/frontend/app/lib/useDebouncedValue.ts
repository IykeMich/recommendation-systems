import { useEffect, useState } from "react";

/** Returns `value` once it has stopped changing for `delayMs`. */
export function useDebouncedValue<Value>(value: Value, delayMs = 250): Value {
  const [debouncedValue, setDebouncedValue] = useState(value);

  // Restart the timer on every change; only the last value survives a full `delayMs` of quiet.
  useEffect(() => {
    const timeoutId = setTimeout(() => setDebouncedValue(value), delayMs);
    return () => clearTimeout(timeoutId);
  }, [value, delayMs]);

  return debouncedValue;
}
