interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

const sizes = {
  sm: "h-4 w-4 border-2",
  md: "h-8 w-8 border-2",
  lg: "h-12 w-12 border-3",
};

export default function Spinner({ size = "md", className = "" }: SpinnerProps) {
  return (
    <div className={`flex items-center justify-center ${className}`} role="status" aria-label="Loading">
      <div
        className={`animate-spin rounded-full border-[var(--line)] border-t-[var(--accent)] ${sizes[size]}`}
      />
      <span className="sr-only">Loading...</span>
    </div>
  );
}
