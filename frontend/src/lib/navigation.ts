import { useSyncExternalStore } from "react";

const navigationEvent = "mi:navigate";

function subscribe(onStoreChange: () => void): () => void {
  window.addEventListener("popstate", onStoreChange);
  window.addEventListener(navigationEvent, onStoreChange);
  return () => {
    window.removeEventListener("popstate", onStoreChange);
    window.removeEventListener(navigationEvent, onStoreChange);
  };
}

function currentPathname(): string {
  return window.location.pathname;
}

export function usePathname(): string {
  return useSyncExternalStore(subscribe, currentPathname, currentPathname);
}

export function navigate(href: string, options: { replace?: boolean } = {}): void {
  const destination = new URL(href, window.location.href);
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  const next = `${destination.pathname}${destination.search}${destination.hash}`;
  if (destination.origin !== window.location.origin) {
    window.location.assign(destination.href);
    return;
  }
  if (current === next) return;
  window.history[options.replace ? "replaceState" : "pushState"]({}, "", next);
  window.dispatchEvent(new Event(navigationEvent));
  window.scrollTo({ top: 0, left: 0 });
}
