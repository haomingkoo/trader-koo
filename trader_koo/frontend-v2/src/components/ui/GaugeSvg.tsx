type GaugeZone = [number, number, string, string];

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

interface GaugeSvgProps {
  value: number;
  zones: ReadonlyArray<GaugeZone>;
  max: number;
  valueLabel: string;
  /** Color for the big value text. Defaults to the active zone color. */
  valueColor?: string;
}

export default function GaugeSvg({
  value,
  zones,
  max,
  valueLabel,
  valueColor,
}: GaugeSvgProps) {
  const cx = 150;
  const cy = 155;
  const r = 120;
  const arcWidth = 14;
  const valueToAngle = (v: number) => 180 + (v / max) * 180;
  const zoneArcs = zones.map(([lo, hi, , color]) => ({
    d: describeArc(cx, cy, r, valueToAngle(lo), valueToAngle(hi)),
    color,
  }));
  const clamped = Math.max(0, Math.min(value, max));
  const needleRotation = valueToAngle(clamped) - 270;
  const currentZone =
    zones.find(([lo, hi]) => value >= lo && value < hi) ?? zones[zones.length - 1];
  const zoneLabel = currentZone[2];
  const zoneColor = currentZone[3];
  const needleLen = r - arcWidth / 2 - 6;

  return (
    <svg viewBox="0 0 300 180" className="w-full max-w-[340px]">
      <path
        d={describeArc(cx, cy, r, 180, 360)}
        fill="none"
        stroke="var(--panel-hover)"
        strokeWidth={arcWidth + 4}
        strokeLinecap="round"
        opacity={0.3}
      />
      {zoneArcs.map((arc, index) => (
        <path
          key={index}
          d={arc.d}
          fill="none"
          stroke={arc.color}
          strokeWidth={arcWidth}
          strokeLinecap="butt"
        />
      ))}
      {(() => {
        const leftCap = polarToCartesian(cx, cy, r, 180);
        const rightCap = polarToCartesian(cx, cy, r, 360);
        return (
          <>
            <circle cx={leftCap.x} cy={leftCap.y} r={arcWidth / 2} fill={zones[0][3]} />
            <circle
              cx={rightCap.x}
              cy={rightCap.y}
              r={arcWidth / 2}
              fill={zones[zones.length - 1][3]}
            />
          </>
        );
      })()}
      <text x={cx - r} y={cy + 18} textAnchor="middle" fontSize={9} fill="var(--muted)" opacity={0.6}>
        0
      </text>
      <text x={cx + r} y={cy + 18} textAnchor="middle" fontSize={9} fill="var(--muted)" opacity={0.6}>
        {max}
      </text>
      <line
        x1={cx}
        y1={cy}
        x2={cx}
        y2={cy - needleLen}
        stroke="var(--text)"
        strokeWidth={2}
        strokeLinecap="round"
        transform={`rotate(${needleRotation}, ${cx}, ${cy})`}
      />
      <circle cx={cx} cy={cy} r={6} fill="var(--panel-hover)" />
      <circle cx={cx} cy={cy} r={3} fill="var(--text)" />
      <text
        x={cx}
        y={cy - 24}
        textAnchor="middle"
        fontSize={34}
        fontWeight={800}
        fill={valueColor ?? zoneColor}
        style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
      >
        {valueLabel}
      </text>
      <text
        x={cx}
        y={cy - 2}
        textAnchor="middle"
        fontSize={11}
        fontWeight={700}
        fill={zoneColor}
        style={{ textTransform: "uppercase", letterSpacing: "0.12em" }}
      >
        {zoneLabel}
      </text>
    </svg>
  );
}
