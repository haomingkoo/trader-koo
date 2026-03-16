import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 pt-24 text-center">
      <div className="text-6xl font-bold text-[var(--line)]">404</div>
      <div className="text-lg text-[var(--muted)]">Page not found</div>
      <Link
        to="/v2"
        className="rounded-lg bg-[var(--accent)] px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-[var(--blue)]"
      >
        Back to Dashboard
      </Link>
    </div>
  );
}
