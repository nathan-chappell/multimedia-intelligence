import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  TransientStatusContext,
  type ShowStatusOptions,
  type StatusTone,
} from "./transientStatus";

interface StatusMessage {
  id: number;
  text: string;
  tone: StatusTone;
}

export function TransientStatusProvider({ children }: { children: ReactNode }) {
  const [message, setMessage] = useState<StatusMessage | null>(null);
  const nextId = useRef(0);
  const timeout = useRef<number | undefined>(undefined);

  const clearTimer = useCallback(() => {
    if (timeout.current !== undefined) window.clearTimeout(timeout.current);
    timeout.current = undefined;
  }, []);

  useEffect(() => clearTimer, [clearTimer]);

  const showStatus = useCallback(
    (text: string, options: ShowStatusOptions = {}) => {
      clearTimer();
      const id = ++nextId.current;
      setMessage({ id, text, tone: options.tone ?? "working" });
      const durationMs = options.durationMs ?? 2600;
      if (durationMs > 0) {
        timeout.current = window.setTimeout(() => {
          setMessage((current) => (current?.id === id ? null : current));
          timeout.current = undefined;
        }, durationMs);
      }
    },
    [clearTimer],
  );

  const value = useMemo(() => ({ showStatus }), [showStatus]);

  return (
    <TransientStatusContext.Provider value={value}>
      {children}
      <div className="transient-status-region" aria-live="polite" aria-atomic="true">
        {message && (
          <div className={`transient-status transient-status-${message.tone}`} role="status">
            <span aria-hidden="true" />
            {message.text}
          </div>
        )}
      </div>
    </TransientStatusContext.Provider>
  );
}
