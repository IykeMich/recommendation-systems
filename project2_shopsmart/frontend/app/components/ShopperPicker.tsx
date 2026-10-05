"use client";

import { useState, type FormEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useDebouncedValue } from "../lib/useDebouncedValue";

// Quick-pick shoppers; GUEST-001 has no history, demonstrating the cold-start fallback.
const SAMPLE_SHOPPERS = [
  { userId: "R00001", label: "R00001" },
  { userId: "R00058", label: "R00058" },
  { userId: "R00500", label: "R00500" },
  { userId: "GUEST-001", label: "New guest" },
];

/**
 * Search box with debounced suggestions, plus sample-shopper pills. Pressing Enter selects any
 * typed ID (upper-cased), even one the API has never seen.
 */
export function ShopperPicker({
  selectedUserId,
  onSelectUser,
}: {
  selectedUserId: string;
  onSelectUser: (userId: string) => void;
}) {
  const [searchInput, setSearchInput] = useState("");
  const [isSuggestionListOpen, setIsSuggestionListOpen] = useState(false);
  const debouncedSearchTerm = useDebouncedValue(searchInput.trim());

  // Search runs only while the list is open and the debounced term is non-empty. keepPreviousData
  // keeps the old suggestions visible while the next term loads, avoiding flicker.
  const shopperSearchQuery = useQuery({
    queryKey: ["shopper-search", debouncedSearchTerm],
    queryFn: () => api.searchShoppers(debouncedSearchTerm),
    enabled: isSuggestionListOpen && debouncedSearchTerm.length > 0,
    placeholderData: keepPreviousData,
  });

  /** Select a shopper and reset the search box. */
  function chooseShopper(userId: string) {
    onSelectUser(userId);
    setSearchInput("");
    setIsSuggestionListOpen(false);
  }

  /** Enter key: use whatever was typed as the user ID. */
  function submitTypedUserId(submitEvent: FormEvent) {
    submitEvent.preventDefault();
    const typedUserId = searchInput.trim().toUpperCase();
    if (typedUserId) chooseShopper(typedUserId);
  }

  const suggestions = shopperSearchQuery.data?.users ?? [];
  const showSuggestions = isSuggestionListOpen && searchInput.trim().length > 0;

  return (
    <div className="shopper-picker">
      <form className="search-box" onSubmit={submitTypedUserId} role="search">
        <label htmlFor="shopper-search" className="visually-hidden">
          Find a shopper
        </label>
        <input
          id="shopper-search"
          className="search-input"
          placeholder="Search shoppers by ID, segment, category or region… (Enter to use any ID)"
          value={searchInput}
          onChange={(changeEvent) => {
            setSearchInput(changeEvent.target.value);
            setIsSuggestionListOpen(true);
          }}
          onFocus={() => setIsSuggestionListOpen(true)}
          // Delay closing so a click on a suggestion can land before the list disappears.
          onBlur={() => setTimeout(() => setIsSuggestionListOpen(false), 150)}
          autoComplete="off"
          spellCheck={false}
          role="combobox"
          aria-expanded={showSuggestions}
          aria-controls="shopper-suggestions"
        />
        {showSuggestions && (
          <ul id="shopper-suggestions" className="suggestions" role="listbox">
            {suggestions.map((shopper) => (
              <li key={shopper.user_id} role="option" aria-selected={false}>
                <button
                  type="button"
                  className="suggestion"
                  // Prevent the input from losing focus (and closing the list) on mouse down.
                  onMouseDown={(mouseEvent) => mouseEvent.preventDefault()}
                  onClick={() => chooseShopper(shopper.user_id)}
                >
                  <strong>{shopper.user_id}</strong>
                  <span className="muted">
                    {shopper.segment} · likes {shopper.preferred_category} · {shopper.region}
                  </span>
                </button>
              </li>
            ))}
            {shopperSearchQuery.isSuccess && suggestions.length === 0 && (
              <li className="suggestion-empty muted">
                No matching shoppers. Press Enter to try “{searchInput.trim().toUpperCase()}”
                as a new shopper.
              </li>
            )}
          </ul>
        )}
      </form>

      <div className="sample-shoppers" aria-label="Sample shoppers">
        <span className="muted small">Try:</span>
        {SAMPLE_SHOPPERS.map((sampleShopper) => (
          <button
            key={sampleShopper.userId}
            type="button"
            className="pill"
            aria-pressed={selectedUserId === sampleShopper.userId}
            onClick={() => chooseShopper(sampleShopper.userId)}
          >
            {sampleShopper.label}
          </button>
        ))}
      </div>
    </div>
  );
}
