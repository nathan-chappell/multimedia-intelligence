import { createContext, useContext } from "react";

export type StatusTone = "working" | "success" | "error";

export interface ShowStatusOptions {
  tone?: StatusTone;
  durationMs?: number;
}

export interface TransientStatusValue {
  showStatus: (text: string, options?: ShowStatusOptions) => void;
}

export const TransientStatusContext = createContext<TransientStatusValue | null>(null);

export function useTransientStatus(): TransientStatusValue {
  const value = useContext(TransientStatusContext);
  if (!value) {
    throw new Error("useTransientStatus must be used inside TransientStatusProvider");
  }
  return value;
}
