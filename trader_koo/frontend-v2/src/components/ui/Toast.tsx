import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { X, TrendingUp, BarChart3, Bitcoin } from "lucide-react";
import type { AlertItem } from "../../api/types";

interface ToastProps {
  alert: AlertItem | null;
  onDismiss: () => void;
}

const SEVERITY_BORDER: Record<string, string> = {
  high: "border-l-[var(--red)]",
  medium: "border-l-[var(--amber)]",
  low: "border-l-[var(--green)]",
};

const TYPE_ICON: Record<string, typeof TrendingUp> = {
  price_alert: TrendingUp,
  market_spike: BarChart3,
  crypto_spike: Bitcoin,
};

export default function Toast({ alert, onDismiss }: ToastProps) {
  const navigate = useNavigate();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (alert) {
      // Trigger slide-in
      requestAnimationFrame(() => setVisible(true));

      const timer = setTimeout(() => {
        setVisible(false);
        setTimeout(onDismiss, 300);
      }, 8000);

      return () => clearTimeout(timer);
    }
    setVisible(false);
    return undefined;
  }, [alert, onDismiss]);

  const handleClick = useCallback(() => {
    setVisible(false);
    setTimeout(() => {
      onDismiss();
      navigate("/alerts");
    }, 150);
  }, [onDismiss, navigate]);

  const handleDismiss = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation();
      setVisible(false);
      setTimeout(onDismiss, 300);
    },
    [onDismiss],
  );

  if (!alert) return null;

  const Icon = TYPE_ICON[alert.type] ?? TrendingUp;
  const borderClass = SEVERITY_BORDER[alert.severity] ?? SEVERITY_BORDER.low;

  return (
    <div
      className={`fixed right-4 top-4 z-[9999] max-w-sm cursor-pointer transition-all duration-300 ease-out ${
        visible
          ? "translate-x-0 opacity-100"
          : "translate-x-full opacity-0"
      }`}
      onClick={handleClick}
      role="alert"
    >
      <div
        className={`rounded-lg border border-[var(--line)] border-l-4 ${borderClass} bg-[var(--panel)] p-3 shadow-lg backdrop-blur-sm`}
      >
        <div className="flex items-start gap-3">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-[var(--panel-hover)]">
            <Icon size={14} className="text-[var(--accent)]" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-semibold text-[var(--text)]">
              {alert.title}
            </p>
            <p className="mt-0.5 truncate text-[10px] text-[var(--muted)]">
              {alert.message}
            </p>
          </div>
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 rounded p-0.5 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
            aria-label="Dismiss notification"
          >
            <X size={12} />
          </button>
        </div>
      </div>
    </div>
  );
}
