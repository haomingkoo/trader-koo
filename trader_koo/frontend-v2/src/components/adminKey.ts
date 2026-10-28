/** Older builds persisted the admin key here. It now lives in React state only. */
export const LEGACY_ADMIN_KEY_STORAGE = "trader_koo_admin_key";

type RemovableStorage = Pick<Storage, "removeItem">;

export function clearLegacyAdminKey(
  getStorage: () => RemovableStorage | undefined = () => globalThis.localStorage,
): void {
  try {
    getStorage()?.removeItem(LEGACY_ADMIN_KEY_STORAGE);
  } catch {
    /* Storage is unavailable, so there is nothing this page can clear. */
  }
}
