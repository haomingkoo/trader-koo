import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { LEGACY_ADMIN_KEY_STORAGE, clearLegacyAdminKey } from "./adminKey.ts";

function memoryStorage(initial: Record<string, string>) {
  const values = new Map(Object.entries(initial));
  return {
    values,
    removeItem: (key: string) => {
      values.delete(key);
    },
  };
}

test("removes an admin key left in browser storage by older builds", () => {
  const storage = memoryStorage({ [LEGACY_ADMIN_KEY_STORAGE]: "old-key", theme: "dark" });

  clearLegacyAdminKey(() => storage);

  assert.equal(storage.values.has(LEGACY_ADMIN_KEY_STORAGE), false);
  assert.equal(storage.values.get("theme"), "dark");
});

test("does not throw when browser storage is blocked", () => {
  assert.doesNotThrow(() =>
    clearLegacyAdminKey(() => {
      throw new Error("SecurityError");
    }),
  );
  assert.doesNotThrow(() =>
    clearLegacyAdminKey(() => ({
      removeItem: () => {
        throw new Error("SecurityError");
      },
    })),
  );
});

test("does not throw when browser storage is absent", () => {
  assert.doesNotThrow(() => clearLegacyAdminKey(() => undefined));
});

test("the app entry clears the legacy key before the first render", () => {
  // The Guide admin panel is not mounted on most visits, so the clear runs at start-up.
  const entry = readFileSync(new URL("../main.tsx", import.meta.url), "utf8");
  const clearAt = entry.indexOf("clearLegacyAdminKey();");

  assert.ok(clearAt >= 0, "main.tsx must call clearLegacyAdminKey()");
  assert.ok(clearAt < entry.indexOf("createRoot("), "clear must run before render");
});
