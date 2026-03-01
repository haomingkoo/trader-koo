from __future__ import annotations

import html
import os
from typing import Any

from trader_koo.email_chart_preview import build_chart_preview_url, chart_preview_enabled


def report_email_app_url() -> str | None:
    for key in ("TRADER_KOO_EMAIL_REPORT_URL", "TRADER_KOO_ALLOWED_ORIGIN"):
        value = str(os.getenv(key, "") or "").strip()
        if value and value != "*" and value.startswith(("http://", "https://")):
            return value.rstrip("/")
    return None


def build_report_email_subject(report: dict[str, Any]) -> str:
    meta = report.get("meta") or {}
    report_kind = "weekly" if str(meta.get("report_kind") or "").strip().lower() == "weekly" else "daily"
    cadence_label = "WEEKLY" if report_kind == "weekly" else "DAILY"
    ok = bool(report.get("ok"))
    status = "OK" if ok else "WARN"
    generated = str(report.get("generated_ts") or "unknown")
    yolo = report.get("yolo") if isinstance(report.get("yolo"), dict) else {}
    primary_delta = yolo.get("delta_weekly") if report_kind == "weekly" else yolo.get("delta_daily")
    if not isinstance(primary_delta, dict):
        primary_delta = {}
    new_count = int(primary_delta.get("new_count") or 0)
    lost_count = int(primary_delta.get("lost_count") or 0)
    primary_tf = "weekly" if report_kind == "weekly" else "daily"
    return f"[trader_koo] {cadence_label} {status} | {generated[:10]} | +{new_count} new -{lost_count} lost {primary_tf} patterns"


def build_report_email_bodies(
    report: dict[str, Any],
    md_text: str,
    *,
    app_url: str | None = None,
) -> tuple[str, str]:
    generated = str(report.get("generated_ts") or "unknown")
    ok = bool(report.get("ok"))
    status = "OK" if ok else "WARN"
    status_color = "#0f9d58" if ok else "#d93025"
    meta = report.get("meta") or {}
    report_kind = "weekly" if str(meta.get("report_kind") or "").strip().lower() == "weekly" else "daily"
    cadence_label = "Weekly" if report_kind == "weekly" else "Daily"
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    latest = report.get("latest_data") if isinstance(report.get("latest_data"), dict) else {}
    freshness = report.get("freshness") if isinstance(report.get("freshness"), dict) else {}
    yolo = report.get("yolo") if isinstance(report.get("yolo"), dict) else {}
    yolo_summary = yolo.get("summary") if isinstance(yolo.get("summary"), dict) else {}
    signals = report.get("signals") if isinstance(report.get("signals"), dict) else {}
    breadth = signals.get("market_breadth") if isinstance(signals.get("market_breadth"), dict) else {}
    vol_ctx = signals.get("volatility_context") if isinstance(signals.get("volatility_context"), dict) else {}
    key_changes = signals.get("tonight_key_changes") if isinstance(signals.get("tonight_key_changes"), list) else []
    movers_up = signals.get("movers_up_today") if isinstance(signals.get("movers_up_today"), list) else []
    movers_down = signals.get("movers_down_today") if isinstance(signals.get("movers_down_today"), list) else []
    setup_rows = signals.get("setup_quality_top") if isinstance(signals.get("setup_quality_top"), list) else []
    sector_rows = signals.get("sector_heatmap") if isinstance(signals.get("sector_heatmap"), list) else []
    earnings = signals.get("earnings_catalysts") if isinstance(signals.get("earnings_catalysts"), dict) else {}
    earnings_rows = earnings.get("rows") if isinstance(earnings.get("rows"), list) else []
    earnings_groups = earnings.get("groups") if isinstance(earnings.get("groups"), list) else []
    risk_filters = report.get("risk_filters") if isinstance(report.get("risk_filters"), dict) else {}
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    session = report.get("market_session") if isinstance(report.get("market_session"), dict) else {}
    daily_delta = yolo.get("delta_daily") if isinstance(yolo.get("delta_daily"), dict) else {}
    weekly_delta = yolo.get("delta_weekly") if isinstance(yolo.get("delta_weekly"), dict) else {}

    primary_delta = weekly_delta if report_kind == "weekly" else daily_delta
    top_setup = setup_rows[0] if setup_rows else {}
    setup_cluster = _setup_cluster(setup_rows)
    cluster_summary = ", ".join(
        f"{row.get('ticker')} {_fmt_num(row.get('score'))} ({row.get('setup_tier') or '-'})"
        for row in setup_cluster[:3]
    )
    top_gainer = movers_up[0] if movers_up else {}
    top_loser = movers_down[0] if movers_down else {}
    breadth_adv = breadth.get("pct_advancing")
    breadth_line = (
        f"{_fmt_num(breadth_adv)}% advancers ({_fmt_num(breadth.get('advancers'))} up / {_fmt_num(breadth.get('decliners'))} down)"
        if breadth
        else "-"
    )
    vix_line = (
        f"VIX {_fmt_num(vol_ctx.get('vix_close'))} ({_fmt_num(vol_ctx.get('vix_percentile_1y'))} pctile)"
        if vol_ctx and vol_ctx.get("vix_close") is not None
        else "-"
    )
    daily_delta_line = _format_delta_line(daily_delta, "Daily")
    weekly_delta_line = _format_delta_line(weekly_delta, "Weekly")
    next_catalyst_line = _format_catalyst_line(earnings_rows[:4])

    text_lines = [
        f"trader_koo {report_kind} brief — {generated}",
        f"Status: {status}",
        "",
        f"Breadth: {breadth_line}",
        f"Volatility: {vix_line}",
        daily_delta_line,
        weekly_delta_line,
    ]
    if next_catalyst_line:
        text_lines.append(next_catalyst_line)
    if top_gainer:
        text_lines.append(
            f"Top gainer: {top_gainer.get('ticker')} {_fmt_signed_pct(top_gainer.get('pct_change'))}"
        )
    if top_loser:
        text_lines.append(
            f"Top loser: {top_loser.get('ticker')} {_fmt_signed_pct(top_loser.get('pct_change'))}"
        )
    if top_setup:
        text_lines.append(
            (
                f"Top setup cluster: {cluster_summary}. "
                if len(setup_cluster) > 1
                else "Top setup: "
            )
            + f"Highest-rated: {top_setup.get('ticker')} score {_fmt_num(top_setup.get('score'))} ({top_setup.get('setup_tier') or '-'})"
            + f" — {_setup_observation(top_setup)}"
            + f" | Action: {_setup_action(top_setup)}"
        )
    if key_changes:
        text_lines += ["", "Tonight's key changes:"]
        for idx, item in enumerate(key_changes[:5], start=1):
            text_lines.append(f"{idx}. {item.get('title', 'Change')}: {item.get('detail', '-')}")
    if warnings:
        text_lines += ["", f"Warnings: {', '.join(str(w) for w in warnings)}"]
    if risk_filters.get("trade_mode") and risk_filters.get("trade_mode") != "normal":
        text_lines += ["", f"Trade mode: {risk_filters.get('trade_mode')}"]
    if session.get("is_holiday"):
        text_lines += ["", f"Market holiday: {session.get('holiday_name') or '-'}"]
    if app_url:
        text_lines += ["", f"Dashboard: {app_url}"]
    text_lines += ["", "Use the dashboard for the full report and chart context."]
    text_body = "\n".join(text_lines)

    cards = [
        ("Tracked", _fmt_num(counts.get("tracked_tickers")), "tickers"),
        ("Price Date", _fmt_text(latest.get("price_date")), "latest close"),
        ("YOLO", _fmt_num(yolo_summary.get("rows_total")), "pattern rows"),
        ("Breadth", _fmt_text(f"{_fmt_num(breadth_adv)}%" if breadth_adv is not None else "-"), "advancers"),
        ("VIX", _fmt_text(vol_ctx.get("vix_close")), "close"),
        ("Price Age", _fmt_text(_fmt_num(freshness.get("price_age_days"))), "days"),
    ]

    def _card_html(title: str, value: str, note: str) -> str:
        return (
            "<td style=\"padding:8px;vertical-align:top;width:33.33%;\">"
            "<div style=\"border:1px solid #e6ecf5;border-radius:14px;padding:14px 14px 12px;background:#f8fbff;\">"
            f"<div style=\"font-size:11px;line-height:16px;text-transform:uppercase;letter-spacing:0.08em;color:#5f6b7a;\">{_esc(title)}</div>"
            f"<div style=\"margin-top:6px;font-size:26px;line-height:30px;font-weight:700;color:#0f172a;\">{_esc(value)}</div>"
            f"<div style=\"margin-top:4px;font-size:12px;line-height:18px;color:#6b7280;\">{_esc(note)}</div>"
            "</div>"
            "</td>"
        )

    cards_html = []
    for idx in range(0, len(cards), 3):
        row = cards[idx:idx + 3]
        cards_html.append("<tr>" + "".join(_card_html(*card) for card in row) + "</tr>")

    key_changes_html = "".join(
        (
            "<li style=\"margin:0 0 10px 0;\">"
            f"<strong>{_esc(item.get('title', 'Change'))}.</strong> {_esc(item.get('detail', '-'))}"
            "</li>"
        )
        for item in key_changes[:5]
    ) or "<li style=\"margin:0;\">No material change summary was generated for this run.</li>"

    def _mover_rows(rows: list[dict[str, Any]], near_key: str) -> str:
        out = []
        for row in rows[:5]:
            move_color = _signed_color(row.get("pct_change"))
            out.append(
                "<tr>"
                f"<td style=\"padding:8px 0;border-bottom:1px solid #eef2f7;font-weight:700;color:#0f172a;\">{_esc(row.get('ticker'))}</td>"
                f"<td style=\"padding:8px 0;border-bottom:1px solid #eef2f7;text-align:right;color:{move_color};font-weight:700;\">{_esc(_fmt_signed_pct(row.get('pct_change')))}</td>"
                f"<td style=\"padding:8px 0 8px 12px;border-bottom:1px solid #eef2f7;color:#6b7280;\">{_esc('near 52W ' + ('high' if near_key.endswith('high') else 'low') if row.get(near_key) else '-')}</td>"
                "</tr>"
            )
        if not out:
            out.append("<tr><td colspan=\"3\" style=\"padding:8px 0;color:#6b7280;\">No data</td></tr>")
        return "".join(out)

    delta_cards_html = "".join(
        _delta_card_html("Daily YOLO", daily_delta)
        + _delta_card_html("Weekly YOLO", weekly_delta)
    )
    catalyst_board_html = _catalyst_board_html(earnings_groups)

    preview_candidates = [
        row
        for row in setup_rows
        if str(row.get("actionability") or "").strip() in {"higher-probability", "conditional"}
        and str(row.get("yolo_pattern") or "").strip()
    ]
    if not preview_candidates:
        preview_candidates = [row for row in setup_rows if str(row.get("yolo_pattern") or "").strip()]
    if not preview_candidates:
        preview_candidates = setup_rows[:]
    preview_rows = preview_candidates[:4]
    previews_enabled = chart_preview_enabled(app_url)
    preview_cards_html = ""
    if previews_enabled and preview_rows:
        cards_out = []
        for row in preview_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            timeframe = str(row.get("yolo_timeframe") or ("weekly" if report_kind == "weekly" else "daily")).strip().lower()
            preview_url = build_chart_preview_url(
                base_url=str(app_url or ""),
                ticker=ticker,
                timeframe=timeframe,
                report_ts=generated,
            )
            if not preview_url:
                continue
            cards_out.append(
                "<td style=\"padding:8px;vertical-align:top;width:50%;\">"
                "<div style=\"border:1px solid #e6ecf5;border-radius:18px;padding:14px;background:#ffffff;\">"
                f"<div style=\"font-size:15px;line-height:20px;font-weight:800;color:#0f172a;\">{_esc(ticker)}</div>"
                f"<div style=\"margin-top:2px;font-size:12px;line-height:18px;color:#64748b;\">{_esc(timeframe.upper())} setup • {_esc(str(row.get('signal_bias') or 'neutral').upper())}</div>"
                f"<img src=\"{_esc(preview_url)}\" alt=\"{_esc(ticker)} chart preview\" style=\"display:block;width:100%;height:auto;margin-top:12px;border-radius:14px;border:1px solid #e6ecf5;background:#0b1220;\" />"
                f"<div style=\"margin-top:10px;font-size:13px;line-height:19px;color:#334155;\"><strong>Read:</strong> {_esc(_setup_observation(row))}</div>"
                f"<div style=\"margin-top:6px;font-size:13px;line-height:19px;color:#475569;\"><strong>Action:</strong> {_esc(_setup_action(row))}</div>"
                "</div>"
                "</td>"
            )
        preview_rows_html = []
        for idx in range(0, len(cards_out), 2):
            preview_rows_html.append("<tr>" + "".join(cards_out[idx:idx + 2]) + "</tr>")
        preview_cards_html = "".join(preview_rows_html)

    setup_table_rows = []
    for row in setup_rows[:6]:
        setup_table_rows.append(
            "<tr>"
            f"<td style=\"padding:10px 0;border-bottom:1px solid #eef2f7;font-weight:700;color:#0f172a;\">{_esc(row.get('ticker'))}</td>"
            f"<td style=\"padding:10px 0;border-bottom:1px solid #eef2f7;text-align:right;color:#0f172a;\">{_esc(_fmt_num(row.get('score')))}</td>"
            f"<td style=\"padding:10px 0 10px 12px;border-bottom:1px solid #eef2f7;color:#334155;\">"
            f"<div><strong>Read:</strong> {_esc(_setup_observation(row))}</div>"
            f"<div style=\"margin-top:4px;color:#475569;\"><strong>Action:</strong> {_esc(_setup_action(row))}</div>"
            f"</td>"
            "</tr>"
        )
    setup_rows_html = "".join(setup_table_rows) or (
        "<tr><td colspan=\"3\" style=\"padding:10px 0;color:#6b7280;\">No setup candidates</td></tr>"
    )

    sector_table_rows = []
    for row in sector_rows[:6]:
        move_color = _signed_color(row.get("avg_pct_change"))
        sector_table_rows.append(
            "<tr>"
            f"<td style=\"padding:8px 0;border-bottom:1px solid #eef2f7;color:#0f172a;\">{_esc(row.get('sector'))}</td>"
            f"<td style=\"padding:8px 0;border-bottom:1px solid #eef2f7;text-align:right;color:{move_color};font-weight:700;\">{_esc(_fmt_signed_pct(row.get('avg_pct_change')))}</td>"
            f"<td style=\"padding:8px 0 8px 12px;border-bottom:1px solid #eef2f7;color:#6b7280;\">{_esc(_fmt_num(row.get('pct_advancing')))}% adv.</td>"
            "</tr>"
        )
    sector_rows_html = "".join(sector_table_rows) or (
        "<tr><td colspan=\"3\" style=\"padding:8px 0;color:#6b7280;\">No sector data</td></tr>"
    )

    next_session_bits = []
    if isinstance(session.get("next_holiday"), dict):
        next_session_bits.append(
            f"Next holiday: {_fmt_text(session['next_holiday'].get('date'))} {_fmt_text(session['next_holiday'].get('name'))}"
        )
    if isinstance(session.get("next_early_close"), dict):
        next_session_bits.append(
            f"Next early close: {_fmt_text(session['next_early_close'].get('date'))} {_fmt_text(session['next_early_close'].get('name'))}"
        )

    app_link_html = (
        f"<a href=\"{_esc(app_url)}\" style=\"display:inline-block;background:#1677ff;color:#ffffff;text-decoration:none;font-weight:700;padding:12px 18px;border-radius:999px;\">Open Dashboard</a>"
        if app_url
        else ""
    )

    html_body = f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#eef3f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;">
    <div style="max-width:760px;margin:0 auto;padding:24px 12px;">
      <div style="background:#ffffff;border:1px solid #dde7f0;border-radius:22px;overflow:hidden;box-shadow:0 10px 30px rgba(15,23,42,0.08);">
        <div style="padding:28px 28px 22px;background:linear-gradient(135deg,#0f172a 0%,#102a56 100%);color:#ffffff;">
          <div style="font-size:13px;line-height:18px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8;">trader_koo</div>
          <div style="margin-top:8px;font-size:30px;line-height:36px;font-weight:800;">{_esc(cadence_label)} Market Brief</div>
          <div style="margin-top:10px;font-size:15px;line-height:22px;opacity:0.92;">
            Generated { _esc(generated) } • Latest price date { _esc(_fmt_text(latest.get("price_date"))) } • { _esc(vix_line) }
          </div>
          <div style="margin-top:18px;">
            <span style="display:inline-block;padding:7px 12px;border-radius:999px;background:{status_color};color:#ffffff;font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">{_esc(status)}</span>
            <span style="display:inline-block;padding:7px 12px;border-radius:999px;background:rgba(255,255,255,0.12);color:#ffffff;font-size:12px;font-weight:600;margin-left:8px;">{_esc(breadth_line)}</span>
          </div>
          <div style="margin-top:18px;">{app_link_html}</div>
        </div>

        <div style="padding:20px 20px 8px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
            {''.join(cards_html)}
          </table>
        </div>

        <div style="padding:8px 28px 0;">
          <h2 style="margin:0 0 12px;font-size:18px;line-height:24px;color:#0f172a;">Tonight's Key Changes</h2>
          <ol style="margin:0;padding-left:18px;color:#334155;font-size:14px;line-height:21px;">
            {key_changes_html}
          </ol>
        </div>

        <div style="padding:24px 28px 0;">
          <h2 style="margin:0 0 12px;font-size:18px;line-height:24px;color:#0f172a;">YOLO Watch</h2>
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
            <tr>{delta_cards_html}</tr>
          </table>
        </div>

        {(
            '<div style="padding:24px 28px 0;">'
            '<h2 style="margin:0 0 12px;font-size:18px;line-height:24px;color:#0f172a;">Catalyst Board</h2>'
            '<div style="font-size:13px;line-height:19px;color:#64748b;margin-bottom:12px;">'
            'Upcoming earnings are grouped by day and session. Use the board to plan levels before the event, not to blindly chase the headline.'
            '</div>'
            f'{catalyst_board_html}'
            '</div>'
        ) if catalyst_board_html else ''}

        {(
            '<div style="padding:24px 28px 0;">'
            '<h2 style="margin:0 0 12px;font-size:18px;line-height:24px;color:#0f172a;">Setup Charts</h2>'
            '<div style="font-size:13px;line-height:19px;color:#64748b;margin-bottom:12px;">Remote images may be hidden until your email client loads them. Each preview shows price, support/resistance, the latest YOLO box, and the upcoming earnings marker when one is available.</div>'
            '<table role="presentation" width="100%" cellspacing="0" cellpadding="0">'
            f'{preview_cards_html}'
            '</table>'
            '</div>'
        ) if preview_cards_html else ''}

        <div style="padding:24px 28px 0;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
            <tr>
              <td style="width:50%;padding-right:10px;vertical-align:top;">
                <div style="border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#ffffff;">
                  <h3 style="margin:0 0 12px;font-size:17px;line-height:22px;color:#0f172a;">Top Gainers</h3>
                  <table role="presentation" width="100%" cellspacing="0" cellpadding="0">{_mover_rows(movers_up, "near_52w_high")}</table>
                </div>
              </td>
              <td style="width:50%;padding-left:10px;vertical-align:top;">
                <div style="border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#ffffff;">
                  <h3 style="margin:0 0 12px;font-size:17px;line-height:22px;color:#0f172a;">Top Losers</h3>
                  <table role="presentation" width="100%" cellspacing="0" cellpadding="0">{_mover_rows(movers_down, "near_52w_low")}</table>
                </div>
              </td>
            </tr>
          </table>
        </div>

        <div style="padding:24px 28px 0;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
            <tr>
              <td style="width:58%;padding-right:10px;vertical-align:top;">
                <div style="border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#ffffff;">
                  <h3 style="margin:0 0 6px;font-size:17px;line-height:22px;color:#0f172a;">Actionable Setup Board</h3>
                  <div style="margin:0 0 12px;font-size:13px;line-height:19px;color:#64748b;">
                    Fresh active YOLO ranks first, then recent persistence, then older context only. Several names can be valid at once; the leader cluster highlights the closest current contenders.
                    {f"<br /><strong>Leader cluster:</strong> {_esc(cluster_summary)}" if len(setup_cluster) > 1 else ""}
                  </div>
                  <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                    <tr>
                      <th align="left" style="padding:0 0 10px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Ticker</th>
                      <th align="right" style="padding:0 0 10px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Score</th>
                      <th align="left" style="padding:0 0 10px 12px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Observation / Action</th>
                    </tr>
                    {setup_rows_html}
                  </table>
                </div>
              </td>
              <td style="width:42%;padding-left:10px;vertical-align:top;">
                <div style="border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#ffffff;">
                  <h3 style="margin:0 0 12px;font-size:17px;line-height:22px;color:#0f172a;">Sector Rotation</h3>
                  <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                    <tr>
                      <th align="left" style="padding:0 0 10px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Sector</th>
                      <th align="right" style="padding:0 0 10px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Avg move</th>
                      <th align="left" style="padding:0 0 10px 12px;color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:0.06em;">Breadth</th>
                    </tr>
                    {sector_rows_html}
                  </table>
                </div>
              </td>
            </tr>
          </table>
        </div>

        <div style="padding:24px 28px 28px;">
          <div style="border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#f8fbff;">
            <div style="font-size:14px;line-height:22px;color:#334155;">
              <strong>How to use this:</strong> Bullish does not mean buy immediately, and bearish does not mean short immediately. The higher-probability cases are where fresh pattern, trend, and level location align. Old YOLO patterns are context only unless current candles and levels re-confirm the idea.<br />
              <strong>Session context:</strong> {_esc(_session_context_line(session))}<br />
              <strong>Risk mode:</strong> {_esc(_risk_mode_line(risk_filters))}<br />
              <strong>Warnings:</strong> {_esc(", ".join(str(w) for w in warnings) if warnings else "none")}<br />
              <strong>Attachments:</strong> full markdown report
            </div>
            <div style="margin-top:14px;font-size:12px;line-height:18px;color:#6b7280;">
              {_esc("This dashboard is for research and education only. It is not financial advice.")}
              {"<br />" + _esc(" • ".join(next_session_bits)) if next_session_bits else ""}
            </div>
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
"""
    return text_body, html_body


def _delta_card_html(title: str, delta: dict[str, Any]) -> str:
    compare = _compare_label(delta)
    highlight = _delta_highlight(delta)
    return (
        "<td style=\"padding:8px;vertical-align:top;width:50%;\">"
        "<div style=\"border:1px solid #e6ecf5;border-radius:18px;padding:18px;background:#ffffff;\">"
        f"<div style=\"font-size:13px;line-height:18px;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;\">{_esc(title)}</div>"
        f"<div style=\"margin-top:8px;font-size:28px;line-height:32px;font-weight:800;color:#0f172a;\">+{_fmt_num(delta.get('new_count') or 0)} / -{_fmt_num(delta.get('lost_count') or 0)}</div>"
        f"<div style=\"margin-top:6px;font-size:13px;line-height:19px;color:#64748b;\">{_esc(compare)}</div>"
        f"<div style=\"margin-top:10px;font-size:14px;line-height:20px;color:#334155;\">{_esc(highlight)}</div>"
        "</div>"
        "</td>"
    )


def _catalyst_board_html(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return ""
    day_blocks = []
    for group in groups[:7]:
        sessions = group.get("sessions") if isinstance(group.get("sessions"), list) else []
        session_rows = []
        for session in sessions:
            rows = session.get("rows") if isinstance(session.get("rows"), list) else []
            if not rows:
                continue
            code = _fmt_text(session.get("code"))
            label = _fmt_text(session.get("label"))
            rendered_rows = []
            for row in rows[:10]:
                rec_state = str(row.get("recommendation_state") or "calendar_only").strip().lower()
                rec_label = {
                    "setup_ready": "SETUP READY",
                    "watch": "WATCH",
                    "calendar_only": "CALENDAR ONLY",
                }.get(rec_state, "CALENDAR ONLY")
                rec_color = {
                    "setup_ready": "#0f9d58",
                    "watch": "#b7791f",
                    "calendar_only": "#64748b",
                }.get(rec_state, "#64748b")
                schedule_label = {
                    "confirmed": "Confirmed",
                    "date_only": "Date only",
                    "snapshot": "Snapshot",
                    "unverified": "Unverified",
                }.get(str(row.get("schedule_quality") or "").strip().lower(), _fmt_text(row.get("schedule_quality")))
                bias_color = _bias_color(row.get("signal_bias"))
                rendered_rows.append(
                    "<tr>"
                    f"<td style=\"padding:8px 0;border-bottom:1px solid #eef2f7;font-weight:700;color:#0f172a;vertical-align:top;\">{_esc(row.get('ticker'))}</td>"
                    f"<td style=\"padding:8px 0 8px 10px;border-bottom:1px solid #eef2f7;text-align:right;color:#0f172a;vertical-align:top;white-space:nowrap;\">{_esc(_fmt_num(row.get('score')))}</td>"
                    f"<td style=\"padding:8px 0 8px 14px;border-bottom:1px solid #eef2f7;color:{bias_color};font-weight:700;vertical-align:top;white-space:nowrap;\">{_esc(str(row.get('signal_bias') or 'neutral').upper())}</td>"
                    f"<td style=\"padding:8px 0 8px 14px;border-bottom:1px solid #eef2f7;color:{_risk_color(row.get('earnings_risk'))};font-weight:700;vertical-align:top;white-space:nowrap;\">{_esc(str(row.get('earnings_risk') or 'normal').upper())}</td>"
                    f"<td style=\"padding:8px 0 8px 14px;border-bottom:1px solid #eef2f7;color:{rec_color};font-weight:700;vertical-align:top;white-space:nowrap;\">{_esc(rec_label)}</td>"
                    f"<td style=\"padding:8px 0 8px 14px;border-bottom:1px solid #eef2f7;color:#475569;vertical-align:top;\">"
                    f"<div style=\"font-weight:600;color:#64748b;\">{_esc(schedule_label)}</div>"
                    f"<div style=\"margin-top:4px;line-height:19px;\">{_esc(str(row.get('recommendation_note') or row.get('action') or '').strip())}</div>"
                    "</td>"
                    "</tr>"
                )
            session_rows.append(
                "<div style=\"margin-top:12px;border-top:1px solid #eef2f7;padding-top:12px;\">"
                f"<div style=\"font-size:12px;line-height:18px;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;\">{_esc(label)} ({_esc(code)})</div>"
                "<table role=\"presentation\" width=\"100%\" cellspacing=\"0\" cellpadding=\"0\" style=\"margin-top:8px;\">"
                + "".join(rendered_rows)
                + "</table></div>"
            )
        if not session_rows:
            continue
        day_blocks.append(
            "<div style=\"margin-top:12px;border:1px solid #e6ecf5;border-radius:18px;padding:16px;background:#ffffff;\">"
            f"<div style=\"font-size:17px;line-height:22px;font-weight:800;color:#0f172a;\">{_esc(group.get('display_date'))}</div>"
            f"<div style=\"margin-top:4px;font-size:12px;line-height:18px;color:#64748b;\">{_esc(_fmt_num(group.get('count')))} earnings events</div>"
            + "".join(session_rows)
            + "</div>"
        )
    return "".join(day_blocks)


def _delta_highlight(delta: dict[str, Any]) -> str:
    new_patterns = delta.get("new_patterns") if isinstance(delta.get("new_patterns"), list) else []
    lost_patterns = delta.get("lost_patterns") if isinstance(delta.get("lost_patterns"), list) else []
    if new_patterns:
        first = new_patterns[0]
        return f"New: {first.get('ticker')} {first.get('pattern')} ({_fmt_num(first.get('confidence'))})"
    if lost_patterns:
        first = lost_patterns[0]
        return f"Lost: {first.get('ticker')} {first.get('pattern')} ({_fmt_num(first.get('confidence'))})"
    return "No material change detected in this comparison window."


def _compare_label(delta: dict[str, Any]) -> str:
    prev_asof = _fmt_text(delta.get("prev_asof"))
    today_asof = _fmt_text(delta.get("today_asof"))
    if prev_asof != "-" and today_asof != "-":
        return f"{prev_asof} -> {today_asof}"
    if int(delta.get("history_retained") or 0) == 1 and today_asof != "-":
        return f"Only one retained snapshot so far ({today_asof})"
    return "Insufficient history for a prior comparison."


def _format_delta_line(delta: dict[str, Any], label: str) -> str:
    if not delta:
        return f"{label} YOLO delta: unavailable"
    return (
        f"{label} YOLO delta: +{_fmt_num(delta.get('new_count') or 0)} / -{_fmt_num(delta.get('lost_count') or 0)} "
        f"({_compare_label(delta)})"
    )


def _format_catalyst_line(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    parts = []
    for row in rows[:4]:
        rec_state = str(row.get("recommendation_state") or "calendar_only").replace("_", " ").upper()
        parts.append(
            f"{_fmt_text(row.get('earnings_date'))} {str(row.get('earnings_session') or 'TBD').upper()} "
            f"{_fmt_text(row.get('ticker'))} ({rec_state}, {str(row.get('earnings_risk') or 'normal').upper()})"
        )
    return "Catalysts: " + " | ".join(parts)


def _risk_mode_line(risk_filters: dict[str, Any]) -> str:
    trade_mode = _fmt_text(risk_filters.get("trade_mode"))
    hard_blocks = _fmt_num(risk_filters.get("hard_blocks") or 0)
    soft_flags = _fmt_num(risk_filters.get("soft_flags") or 0)
    return f"{trade_mode} | hard blocks {hard_blocks} | soft flags {soft_flags}"


def _session_context_line(session: dict[str, Any]) -> str:
    bits = [
        _fmt_text(session.get("market_date")),
        _fmt_text(session.get("market_tz")),
    ]
    if session.get("is_holiday"):
        bits.append(f"holiday: {_fmt_text(session.get('holiday_name'))}")
    if session.get("is_early_close"):
        bits.append(f"early close: {_fmt_text(session.get('early_close_name'))}")
    return " | ".join(bit for bit in bits if bit and bit != "-")


def _setup_reason(row: dict[str, Any]) -> str:
    parts = []
    if row.get("discount_pct") not in {None, ""}:
        parts.append(f"discount {_fmt_num(row.get('discount_pct'))}%")
    if row.get("peg") not in {None, ""}:
        parts.append(f"PEG {_fmt_num(row.get('peg'))}")
    if row.get("pct_change") not in {None, ""}:
        parts.append(f"move {_fmt_signed_pct(row.get('pct_change'))}")
    if row.get("yolo_pattern"):
        conf = row.get("yolo_confidence")
        if conf not in {None, ""}:
            parts.append(f"{row.get('yolo_pattern')} {_fmt_num(conf)}")
        else:
            parts.append(str(row.get("yolo_pattern")))
    if row.get("atr_pct_14") not in {None, ""}:
        parts.append(f"ATR {_fmt_num(row.get('atr_pct_14'))}%")
    return ", ".join(parts[:4]) or "Composite valuation, momentum, volatility, and pattern strength."


def _setup_observation(row: dict[str, Any]) -> str:
    observation = str(row.get("observation") or "").strip()
    if observation:
        return observation
    return _setup_reason(row)


def _setup_action(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "").strip()
    if action:
        return action
    return "Watch only until trend, pattern, and level location line up."


def _setup_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    best = rows[0]
    best_score = float(best.get("score") or 0.0)
    best_tier = str(best.get("setup_tier") or "").strip().upper()
    cluster = [best]
    for row in rows[1:5]:
        row_tier = str(row.get("setup_tier") or "").strip().upper()
        row_score = float(row.get("score") or 0.0)
        if row_tier == best_tier and (best_score - row_score) <= 3.0:
            cluster.append(row)
    return cluster


def _fmt_signed_pct(value: Any) -> str:
    if value in {None, ""}:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if num > 0 else ""
    if abs(num) >= 10:
        return f"{sign}{num:.2f}%"
    return f"{sign}{num:.2f}%"


def _fmt_num(value: Any) -> str:
    if value in {None, ""}:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num.is_integer() and abs(num) >= 1000:
        return f"{int(num):,}"
    if abs(num) >= 100:
        return f"{num:,.1f}"
    if abs(num) >= 10:
        return f"{num:.2f}".rstrip("0").rstrip(".")
    return f"{num:.2f}".rstrip("0").rstrip(".")


def _fmt_text(value: Any) -> str:
    if value in {None, ""}:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _esc(value: Any) -> str:
    return html.escape(_fmt_text(value))


def _risk_color(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "high":
        return "#d93025"
    if raw == "elevated":
        return "#b45309"
    return "#0f766e"


def _bias_color(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "bullish":
        return "#0f766e"
    if raw == "bearish":
        return "#d93025"
    return "#64748b"


def _signed_color(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "#0f172a"
    if num > 0:
        return "#0f9d58"
    if num < 0:
        return "#d93025"
    return "#64748b"
