"use client";

import { useState, type FormEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, type UserProfile } from "../lib/api";
import { useDebouncedValue } from "../lib/useDebouncedValue";

// Quick-pick shoppers; NEW-VISITOR has no profile or history, so it shows the cold-start path.
const SAMPLE_USERS = ["R00001", "R00003", "R00250", "R00777"];
const UNKNOWN_USER_ID = "NEW-VISITOR";

/**
 * Search box with suggestions plus sample-user pills. Enter selects whatever ID was typed,
 * even one that doesn't exist, to demonstrate cold start.
 */
export function UserPicker({
  selectedUserId,
  onSelectUser,
}: {
  selectedUserId: string;
  onSelectUser: (userId: string, profile?: UserProfile) => void;
}) {
  const [searchInput, setSearchInput] = useState("");
  const [isSuggestionListOpen, setIsSuggestionListOpen] = useState(false);
  const debouncedSearchTerm = useDebouncedValue(searchInput.trim());

  // Search runs on the debounced term, only while the suggestion list is open and non-empty.
  const userSearchQuery = useQuery({
    queryKey: ["user-search", debouncedSearchTerm],
    queryFn: () => api.searchUsers(debouncedSearchTerm),
    enabled: isSuggestionListOpen && debouncedSearchTerm.length > 0,
    placeholderData: keepPreviousData,
  });

  function chooseUser(userId: string, profile?: UserProfile) {
    onSelectUser(userId, profile);
    setSearchInput("");
    setIsSuggestionListOpen(false);
  }

  function submitTypedUserId(submitEvent: FormEvent) {
    submitEvent.preventDefault();
    const typedUserId = searchInput.trim().toUpperCase();
    if (typedUserId) chooseUser(typedUserId);
  }

  const suggestions = userSearchQuery.data?.users ?? [];
  const showSuggestions = isSuggestionListOpen && searchInput.trim().length > 0;

  return (
    <div className="user-picker">
      <form className="search-box" onSubmit={submitTypedUserId} role="search">
        <label htmlFor="user-search" className="visually-hidden">
          Find a shopper
        </label>
        <input
          id="user-search"
          className="search-input"
          placeholder="Search shoppers by ID, segment, category or region… (Enter to use any ID)"
          value={searchInput}
          onChange={(changeEvent) => {
            setSearchInput(changeEvent.target.value);
            setIsSuggestionListOpen(true);
          }}
          onFocus={() => setIsSuggestionListOpen(true)}
          // Close after a short delay so a click on a suggestion still registers.
          onBlur={() => setTimeout(() => setIsSuggestionListOpen(false), 150)}
          autoComplete="off"
          spellCheck={false}
          role="combobox"
          aria-expanded={showSuggestions}
          aria-controls="user-suggestions"
        />
        {showSuggestions && (
          <ul id="user-suggestions" className="suggestions" role="listbox">
            {suggestions.map((profile) => (
              <li key={profile.user_id} role="option" aria-selected={false}>
                <button
                  type="button"
                  className="suggestion"
                  // Prevent the input from losing focus (and closing the list) before onClick fires.
                  onMouseDown={(mouseEvent) => mouseEvent.preventDefault()}
                  onClick={() => chooseUser(profile.user_id, profile)}
                >
                  <strong>{profile.user_id}</strong>
                  <span className="muted">
                    {profile.segment} · likes {profile.preferred_category} · {profile.device_type} ·{" "}
                    {profile.region}
                  </span>
                </button>
              </li>
            ))}
            {userSearchQuery.isSuccess && suggestions.length === 0 && (
              <li className="suggestion-empty muted">
                No matching shoppers. Press Enter to try “{searchInput.trim().toUpperCase()}”.
              </li>
            )}
          </ul>
        )}
      </form>

      <div className="sample-users" aria-label="Sample shoppers">
        <span className="muted small">Try:</span>
        {[...SAMPLE_USERS, UNKNOWN_USER_ID].map((sampleUserId) => (
          <button
            key={sampleUserId}
            type="button"
            className="pill"
            aria-pressed={selectedUserId === sampleUserId}
            onClick={() => chooseUser(sampleUserId)}
          >
            {sampleUserId === UNKNOWN_USER_ID ? "Unknown visitor" : sampleUserId}
          </button>
        ))}
      </div>
    </div>
  );
}
