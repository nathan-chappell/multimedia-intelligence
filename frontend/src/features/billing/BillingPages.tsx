import { FormEvent, useCallback, useEffect, useState } from "react";
import { UserButton } from "@clerk/react";

import { useSessionUser, type SessionUser } from "../auth/AuthPages";
import { authenticatedFetch } from "../../lib/config";
import { Link } from "../../lib/Link";

interface LedgerEvent {
  id: string;
  user_id: string;
  amount_microusd: number;
  event_type: string;
  description: string | null;
  actor_user_id: string | null;
  thread_id: string | null;
  provider_request_id: string | null;
  provider_response_id: string | null;
  trace_id: string | null;
  agent_span_id: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

interface LedgerPageResponse {
  balance_microusd: number | null;
  items: LedgerEvent[];
  total: number;
}

interface Coupon {
  id: string;
  code_hint: string;
  clear_code: string | null;
  label: string;
  amount_microusd: number;
  max_redemptions: number;
  redemption_count: number;
  active: boolean;
  expires_at: string | null;
}

export function AccountPage() {
  const { user, reload } = useSessionUser();
  const [ledger, setLedger] = useState<LedgerPageResponse>();
  const [code, setCode] = useState("");
  const [message, setMessage] = useState<string>();

  const load = useCallback(async () => {
    const response = await authenticatedFetch("/api/billing/ledger");
    if (!response.ok) throw new Error(await errorDetail(response));
    setLedger((await response.json()) as LedgerPageResponse);
  }, []);

  useEffect(() => {
    queueMicrotask(() => void load());
  }, [load]);

  async function redeem(event: FormEvent) {
    event.preventDefault();
    setMessage(undefined);
    const response = await authenticatedFetch("/api/billing/coupons/redeem", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    if (!response.ok) {
      setMessage(await errorDetail(response));
      return;
    }
    setCode("");
    setMessage("Coupon redeemed.");
    await Promise.all([load(), reload()]);
  }

  return (
    <BillingShell title="Your account">
      <section className="billing-card balance-card">
        <div className="balance-summary">
          <span className="eyebrow">Available credit</span>
          <strong>{formatUsd(ledger?.balance_microusd ?? user.balance_microusd)}</strong>
        </div>
        <section className="redeem-section" aria-labelledby="redeem-coupon-title">
          <div>
            <h2 id="redeem-coupon-title">Redeem coupon</h2>
            <p>Add test credit with an access code.</p>
          </div>
          <form className="coupon-form" onSubmit={(event) => void redeem(event)}>
            <label className="visually-hidden" htmlFor="coupon-code">Coupon code</label>
            <input
              id="coupon-code"
              value={code}
              onChange={(event) => setCode(event.target.value)}
              placeholder="MI-EARLY-ACCESS"
              autoCapitalize="characters"
              autoComplete="off"
              spellCheck={false}
              required
            />
            <button type="submit">Apply code</button>
          </form>
          {message && <p className="redeem-message" role="status">{message}</p>}
        </section>
      </section>
      <LedgerList items={ledger?.items ?? []} />
    </BillingShell>
  );
}

export function AdminPage() {
  const [users, setUsers] = useState<SessionUser[]>([]);
  const [ledger, setLedger] = useState<LedgerEvent[]>([]);
  const [coupons, setCoupons] = useState<Coupon[]>([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [message, setMessage] = useState<string>();
  const [clearCode, setClearCode] = useState<string>();

  const load = useCallback(async () => {
    const [userResponse, ledgerResponse, couponResponse] = await Promise.all([
      authenticatedFetch("/api/admin/users?limit=50"),
      authenticatedFetch("/api/admin/billing/ledger?limit=100"),
      authenticatedFetch("/api/admin/billing/coupons"),
    ]);
    for (const response of [userResponse, ledgerResponse, couponResponse]) {
      if (!response.ok) throw new Error(await errorDetail(response));
    }
    const userPage = await userResponse.json() as { items: SessionUser[] };
    setUsers(userPage.items);
    setSelectedUser((current) => current || userPage.items[0]?.id || "");
    setLedger(((await ledgerResponse.json()) as LedgerPageResponse).items);
    setCoupons((await couponResponse.json()) as Coupon[]);
  }, []);

  useEffect(() => {
    queueMicrotask(() => {
      void load().catch((error: unknown) => setMessage(String(error)));
    });
  }, [load]);

  async function adjust(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const response = await authenticatedFetch("/api/admin/billing/adjustments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: selectedUser,
        amount_usd: Number(form.get("amount")),
        description: String(form.get("description")),
      }),
    });
    setMessage(response.ok ? "Adjustment appended." : await errorDetail(response));
    if (response.ok) await load();
  }

  async function createCoupon(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const response = await authenticatedFetch("/api/admin/billing/coupons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label: String(form.get("label")),
        code: String(form.get("code") || "") || null,
        amount_usd: Number(form.get("amount")),
        max_redemptions: Number(form.get("cap")),
        expires_at: String(form.get("expires") || "") || null,
      }),
    });
    if (!response.ok) {
      setMessage(await errorDetail(response));
      return;
    }
    const coupon = await response.json() as Coupon;
    setClearCode(coupon.clear_code ?? undefined);
    setMessage("Coupon created. Copy the clear code now; it will not be shown again.");
    await load();
  }

  async function deactivate(couponId: string) {
    const response = await authenticatedFetch(`/api/admin/billing/coupons/${couponId}/deactivate`, { method: "POST" });
    setMessage(response.ok ? "Coupon deactivated." : await errorDetail(response));
    if (response.ok) await load();
  }

  return (
    <BillingShell title="Access and billing admin" compact>
      {message && <p className="billing-message" role="status">{message}</p>}
      {clearCode && <pre className="coupon-code">{clearCode}</pre>}
      <div className="billing-grid">
        <section className="billing-card">
          <h2>User adjustment</h2>
          <form className="stack-form" onSubmit={(event) => void adjust(event)}>
            <select aria-label="User" value={selectedUser} onChange={(event) => setSelectedUser(event.target.value)}>
              {users.map((user) => <option key={user.id} value={user.id}>{user.username} · {formatUsd(user.balance_microusd)}</option>)}
            </select>
            <input aria-label="Signed USD amount" name="amount" type="number" step="0.000001" placeholder="Signed USD amount" required />
            <input aria-label="Reason for adjustment" name="description" placeholder="Reason for adjustment" required />
            <button type="submit">Append adjustment</button>
          </form>
        </section>
        <section className="billing-card">
          <h2>Create campaign coupon</h2>
          <form className="stack-form" onSubmit={(event) => void createCoupon(event)}>
            <input aria-label="Campaign label" name="label" placeholder="Campaign label" required />
            <input aria-label="Optional custom code" name="code" placeholder="Optional custom code" />
            <input aria-label="USD credit" name="amount" type="number" min="0.000001" step="0.000001" placeholder="USD credit" required />
            <input aria-label="Redemption cap" name="cap" type="number" min="1" placeholder="Redemption cap" required />
            <input aria-label="Expiration" name="expires" type="datetime-local" />
            <button type="submit">Create coupon</button>
          </form>
        </section>
      </div>
      <section className="billing-card">
        <h2>Coupon campaigns</h2>
        <div className="coupon-list">
          {coupons.length === 0 && <p className="compact-empty">No campaigns yet.</p>}
          {coupons.map((coupon) => (
            <div key={coupon.id}>
              <strong>{coupon.label}</strong> <code>{coupon.code_hint}</code>
              <span>{formatUsd(coupon.amount_microusd)} · {coupon.redemption_count}/{coupon.max_redemptions}</span>
              {coupon.active && <button type="button" onClick={() => void deactivate(coupon.id)}>Deactivate</button>}
            </div>
          ))}
        </div>
      </section>
      <LedgerList items={ledger} users={users} />
    </BillingShell>
  );
}

function BillingShell({ title, children, compact = false }: { title: string; children: React.ReactNode; compact?: boolean }) {
  return (
    <main className={`billing-page${compact ? " billing-page-compact" : ""}`}>
      <header className="billing-header">
        <div><Link href="/">← Workspace</Link><h1>{title}</h1></div>
        <UserButton />
      </header>
      {children}
    </main>
  );
}

function LedgerList({ items, users = [] }: { items: LedgerEvent[]; users?: SessionUser[] }) {
  const usernames = new Map(users.map((user) => [user.id, user.username]));
  return (
    <section className="billing-card">
      <h2>Immutable ledger</h2>
      {items.length === 0 ? (
        <p className="compact-empty">No ledger events yet.</p>
      ) : (
        <ol className="ledger-list">
          {items.map((item) => (
            <li key={item.id} className="ledger-row">
              <div className="ledger-event">
                {users.length > 0 && (
                  <span className="ledger-user">{usernames.get(item.user_id) ?? shortId(item.user_id)}</span>
                )}
                <strong>{item.description ?? humanizeEvent(item.event_type)}</strong>
                <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString()}</time>
              </div>
              <AttributionPreview item={item} />
              <strong className={`ledger-amount ${item.amount_microusd > 0 ? "credit" : "debit"}`}>
                {formatSignedUsd(item.amount_microusd)}
              </strong>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function AttributionPreview({ item }: { item: LedgerEvent }) {
  const attribution = item.provider_response_id
    ?? item.provider_request_id
    ?? item.agent_span_id
    ?? (item.description ? "manual event" : item.event_type);
  const fields = [
    ["Response", item.provider_response_id],
    ["Request", item.provider_request_id],
    ["Agent span", item.agent_span_id],
    ["Trace", item.trace_id],
    ["Thread", item.thread_id],
  ].filter((field): field is [string, string] => typeof field[1] === "string");

  return (
    <details className="attribution-preview">
      <summary title={attribution}>{shortId(attribution)}</summary>
      <div className="attribution-card">
        <strong>Event attribution</strong>
        {fields.length > 0 ? (
          <dl>{fields.map(([label, value]) => (
            <div key={label}><dt>{label}</dt><dd><code>{value}</code></dd></div>
          ))}</dl>
        ) : <p>This entry was appended without a provider request.</p>}
        {item.metadata && Object.keys(item.metadata).length > 0 && (
          <pre>{JSON.stringify(item.metadata, null, 2)}</pre>
        )}
        {item.provider_response_id && (
          <a
            href="https://developers.openai.com/api/reference/java/resources/responses/methods/retrieve"
            target="_blank"
            rel="noreferrer"
          >
            OpenAI response retrieval reference ↗
          </a>
        )}
      </div>
    </details>
  );
}

function formatUsd(value: number): string {
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 6 }).format(value / 1_000_000);
}

function formatSignedUsd(value: number): string {
  return `${value > 0 ? "+" : "−"}${formatUsd(Math.abs(value))}`;
}

function humanizeEvent(value: string): string {
  const normalized = value.replaceAll("_", " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function shortId(value: string): string {
  return value.length > 24 ? `${value.slice(0, 13)}…${value.slice(-7)}` : value;
}

async function errorDetail(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null) as { detail?: string } | null;
  return payload?.detail ?? `Request failed (${response.status})`;
}
