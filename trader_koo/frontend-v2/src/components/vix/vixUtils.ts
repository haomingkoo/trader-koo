export const formatVixState = (value: string | null | undefined): string =>
  (value ?? "unknown")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
