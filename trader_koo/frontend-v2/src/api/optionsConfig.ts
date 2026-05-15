export const OPTIONS_PREMIUM_CONFIG = {
  endpoint: "/api/options/premium",
  defaultLimit: 100,
  pageLimit: 150,
  defaultSort: "volume_premium",
  staleTimeMs: 120_000,
  emptyMessage: "No option-chain snapshots are available yet.",
  unavailableMessage: "Options premium proxy unavailable.",
  noSmartMatchMessage: "No names match this smart view.",
  positivePremiumFloor: 0,
  defaultSignalLabel: "Watch",
  defaultTagVariant: "muted",
  scoreBands: {
    strong: 75,
    moderate: 60,
  },
  sortLabels: {
    volume_premium: "Volume Premium",
    oi_premium: "OI Premium",
    ticker: "Ticker",
  },
  premiumBiasLabels: {
    call_premium_skew: "Call skew",
    put_premium_skew: "Put skew",
    balanced: "Balanced",
    unknown: "Unknown",
  },
  premiumBiasVariants: {
    call_premium_skew: "green",
    put_premium_skew: "red",
    balanced: "amber",
    unknown: "muted",
  },
  smartViews: {
    best: "Best",
    value: "Value Flow",
    calls: "Call Flow",
    hedge: "Hedge",
    hot: "High IV",
    all: "All",
  },
  defaultSmartView: "best",
  smartViewRules: {
    best: {
      signals: ["bullish_candidate", "relative_value"],
    },
    value: {
      requiredTags: ["relative_value_iv"],
      excludedTags: ["hot_iv"],
    },
    calls: {
      premiumBiases: ["call_premium_skew"],
      netVolumePremium: "positive",
    },
    hedge: {
      premiumBiases: ["put_premium_skew"],
      optionalTags: ["put_oi_heavy"],
    },
    hot: {
      requiredTags: ["hot_iv"],
    },
    all: {},
  },
  smartSignals: {
    bullish_candidate: "Bullish",
    bearish_or_hedge: "Hedge",
    relative_value: "Value",
    momentum_chase: "Chase",
    watch: "Watch",
  },
  tagLabels: {
    top_score: "Top",
    strong_flow: "Flow",
    liquid: "Liquid",
    relative_value_iv: "Value IV",
    hot_iv: "High IV",
    limited_history: "New",
    call_oi_lead: "Call OI",
    put_oi_heavy: "Put OI",
    oi_confirmed: "OI Confirm",
  },
  tagVariants: {
    top_score: "green",
    strong_flow: "amber",
    liquid: "green",
    relative_value_iv: "green",
    hot_iv: "red",
    limited_history: "muted",
    call_oi_lead: "muted",
    put_oi_heavy: "red",
    oi_confirmed: "amber",
  },
} as const;

export type OptionsPremiumSort = keyof typeof OPTIONS_PREMIUM_CONFIG.sortLabels;
export type OptionsSmartView = keyof typeof OPTIONS_PREMIUM_CONFIG.smartViews;
