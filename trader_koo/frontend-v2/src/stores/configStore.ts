import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ConfigState {
  apiBase: string;
  setApiBase: (base: string) => void;
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      apiBase: "",
      setApiBase: (base: string) => set({ apiBase: base }),
    }),
    { name: "trader-koo-config" },
  ),
);
