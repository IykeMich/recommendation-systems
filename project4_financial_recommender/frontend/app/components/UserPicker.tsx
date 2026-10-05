"use client";

import { useState, type FormEvent } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, type UserProfile } from "../lib/api";
import { useDebouncedValue } from "../lib/useDebouncedValue";

/** Hand-picked users that show different eligibility situations. */
const SAMPLE_USERS = [
  { userId: "F00278", label: "F00278 · low risk, low income" },
  { userId: "F00003", label: "F00003 · low risk, high income" },
  { userId: "F00815", label: "F00815 · student, high risk" },
  { userId: "F00074", label: "F00074 · business owner" },
];
/** An ID not in the dataset, to demonstrate cold start. */
const UNKNOWN_USER_ID = "NEW-VISITOR";

/** Search box with live suggestions, Enter-to-load any typed ID, and sample-user pills. */
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

  // Search only while the list is open and there is a (debounced) term; keep old results meanwhile.
  const userSearchQuery = useQuery({
    queryKey: ["user-search", debouncedSearchTerm],
    queryFn: () => api.searchUsers(debouncedSearchTerm),
    enabled: isSuggestionListOpen && debouncedSearchTerm.length > 0,
    placeholderData: keepPreviousData,
  });

  /** Select a user, then clear and close the search box. */
  function chooseUser(userId: string, profile?: UserProfile) {
    onSelectUser(userId, profile);
    setSearchInput("");
    setIsSuggestionListOpen(false);
  }

  /** Enter loads whatever ID was typed (upper-cased), even one the dataset doesn't contain. */
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
          Find a user
        </label>
        <input
          id="user-search"
          className="search-input"
          placeholder="Search users by ID, employment, risk profile, region or age band… (Enter for any ID)"
          value={searchInput}
          onChange={(changeEvent) => {
            setSearchInput(changeEvent.target.value);
            setIsSuggestionListOpen(true);
          }}
          onFocus={() => setIsSuggestionListOpen(true)}
          // Delay closing so a click on a suggestion still registers.
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
                  // Prevent the input losing focus (and the list closing) before onClick runs.
                  onMouseDown={(mouseEvent) => mouseEvent.preventDefault()}
                  onClick={() => chooseUser(profile.user_id, profile)}
                >
                  <strong>{profile.user_id}</strong>
                  <span className="muted">
                    {profile.employment?.replace("_", " ")} · {profile.risk_profile} risk · {profile.age_band} ·{" "}
                    {profile.region}
                  </span>
                </button>
              </li>
            ))}
            {userSearchQuery.isSuccess && suggestions.length === 0 && (
              <li className="suggestion-empty muted">
                No matching users. Press Enter to try “{searchInput.trim().toUpperCase()}”.
              </li>
            )}
          </ul>
        )}
      </form>

      <div className="sample-users" aria-label="Sample users">
        <span className="muted small">Try:</span>
        {[...SAMPLE_USERS, { userId: UNKNOWN_USER_ID, label: "Unknown visitor" }].map(({ userId: sampleUserId, label }) => (
          <button
            key={sampleUserId}
            type="button"
            className="pill"
            aria-pressed={selectedUserId === sampleUserId}
            onClick={() => chooseUser(sampleUserId)}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
