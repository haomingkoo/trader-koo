import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, TrendingUp, BarChart3, Bitcoin } from "lucide-react";
import { useAlerts } from "../../api/hooks";
import Toast from "../ui/Toast";
import type { AlertItem } from "../../api/types";

const LAST_READ_KEY = "trader_koo_last_read_alert_ts";

const TYPE_ICON: Record<string, typeof TrendingUp> = {
  price_alert: TrendingUp,
  market_spike: BarChart3,
  crypto_spike: Bitcoin,
};

const SEVERITY_DOT: Record<string, string> = {
  high: "bg-[var(--red)]",
  medium: "bg-[var(--amber)]",
  low: "bg-[var(--muted)]",
};

function getUnreadCount(alerts: AlertItem[]): number {
  const lastRead = localStorage.getItem(LAST_READ_KEY) ?? "";
  if (!lastRead) return alerts.length;
  return alerts.filter((a) => a.timestamp > lastRead).length;
}

function markAsRead(): void {
  localStorage.setItem(
    LAST_READ_KEY,
    new Date().toISOString(),
  );
}

export default function NotificationBell() {
  const navigate = useNavigate();
  const { data } = useAlerts(5);
  const [open, setOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [toastAlert, setToastAlert] = useState<AlertItem | null>(null);
  const prevCountRef = useRef<number>(0);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const alerts = data?.alerts ?? [];

  // Track unread count
  useEffect(() => {
    const count = getUnreadCount(alerts);
    setUnreadCount(count);
  }, [alerts]);

  // Toast on new alerts
  useEffect(() => {
    const count = getUnreadCount(alerts);
    if (count > prevCountRef.current && prevCountRef.current >= 0 && alerts.length > 0) {
      setToastAlert(alerts[0]);
    }
    prevCountRef.current = count;
  }, [alerts]);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
    return undefined;
  }, [open]);

  const handleBellClick = useCallback(() => {
    setOpen((prev) => {
      if (!prev) {
        // Opening dropdown — mark as read
        markAsRead();
        setUnreadCount(0);
      }
      return !prev;
    });
  }, []);

  const handleViewAll = useCallback(() => {
    setOpen(false);
    navigate("/alerts");
  }, [navigate]);

  const handleAlertClick = useCallback(
    (_alert: AlertItem) => {
      setOpen(false);
      navigate("/alerts");
    },
    [navigate],
  );

  const handleToastDismiss = useCallback(() => {
    setToastAlert(null);
  }, []);

  const hasAlerts = unreadCount > 0;

  return (
    <>
      <div ref={dropdownRef} className="relative">
        <button
          onClick={handleBellClick}
          className={`relative rounded p-1 transition-colors hover:bg-[var(--panel-hover)] ${
            hasAlerts
              ? "text-[var(--accent)]"
              : "text-[var(--muted)] hover:text-[var(--text)]"
          }`}
          aria-label={`Notifications${hasAlerts ? ` (${unreadCount} unread)` : ""}`}
        >
          <Bell size={14} />
          {hasAlerts && (
            <span className="absolute -right-0.5 -top-0.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[var(--red)] text-[8px] font-bold text-white animate-pulse">
              {unreadCount > 9 ? "9+" : unreadCount}
            </span>
          )}
        </button>

        {open && (
          <div className="absolute right-0 top-full z-50 mt-2 w-[360px] overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--panel)] shadow-xl backdrop-blur-sm">
            <div className="border-b border-[var(--line)] px-3 py-2">
              <p className="text-xs font-semibold text-[var(--text)]">
                Notifications
              </p>
            </div>

            {alerts.length === 0 ? (
              <div className="px-3 py-6 text-center">
                <Bell size={20} className="mx-auto mb-2 text-[var(--muted)]" />
                <p className="text-xs text-[var(--muted)]">No recent alerts</p>
              </div>
            ) : (
              <div className="max-h-72 overflow-y-auto">
                {alerts.map((alert) => {
                  const Icon = TYPE_ICON[alert.type] ?? TrendingUp;
                  const dotClass =
                    SEVERITY_DOT[alert.severity] ?? SEVERITY_DOT.low;
                  return (
                    <button
                      key={alert.id}
                      onClick={() => handleAlertClick(alert)}
                      className="flex w-full items-start gap-2.5 border-b border-[var(--line)] px-3 py-2.5 text-left transition-colors hover:bg-[var(--panel-hover)] last:border-b-0"
                    >
                      <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-md bg-[var(--panel-hover)]">
                        <Icon size={12} className="text-[var(--accent)]" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span
                            className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${dotClass}`}
                          />
                          <p className="truncate text-[11px] font-medium text-[var(--text)]">
                            {alert.title}
                          </p>
                        </div>
                        <p className="mt-0.5 truncate text-[10px] text-[var(--muted)]">
                          {alert.message}
                        </p>
                        <p className="mt-0.5 text-[9px] text-[var(--muted)]">
                          {alert.time_ago}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            <div className="border-t border-[var(--line)]">
              <button
                onClick={handleViewAll}
                className="w-full px-3 py-2 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:bg-[var(--panel-hover)]"
              >
                View all alerts
              </button>
            </div>
          </div>
        )}
      </div>

      <Toast alert={toastAlert} onDismiss={handleToastDismiss} />
    </>
  );
}
