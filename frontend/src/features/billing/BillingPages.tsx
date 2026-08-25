import { FormEvent, useCallback, useEffect, useState } from "react";
import { UserButton } from "@clerk/react";

import { useSessionUser, type SessionUser } from "../auth/AuthPages";
import { authenticatedFetch } from "../../lib/config";

interface LedgerEvent {
  id: string;
  user_id: string;
  amount_microusd: number;
  event_type: string;
  description: string | null;
  provider_request_id: string | null;
  agent_span_id: string | null;
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
        <span className="eyebrow">Available credit</span>
        <strong>{formatUsd(ledger?.balance_microusd ?? user.balance_microusd)}</strong>
        <form className="inline-form" onSubmit={(event) => void redeem(event)}>
          <input aria-label="Coupon code" value={code} onChange={(event) => setCode(event.target.value)} placeholder="MI-EARLY-ACCESS" required />
          <button type="submit">Redeem coupon</button>
        </form>
        {message && <p role="status">{message}</p>}
      </section>
      <LedgerTable items={ledger?.items ?? []} showUser={false} />
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
    <BillingShell title="Access and billing admin">
      {message && <p className="billing-message" role="status">{message}</p>}
      {clearCode && <pre className="coupon-code">{clearCode}</pre>}
      <div className="billing-grid">
        <section className="billing-card">
          <h2>User adjustment</h2>
          <form className="stack-form" onSubmit={(event) => void adjust(event)}>
            <select value={selectedUser} onChange={(event) => setSelectedUser(event.target.value)}>
              {users.map((user) => <option key={user.id} value={user.id}>{user.username} · {formatUsd(user.balance_microusd)}</option>)}
            </select>
            <input name="amount" type="number" step="0.000001" placeholder="Signed USD amount" required />
            <input name="description" placeholder="Reason for adjustment" required />
            <button type="submit">Append adjustment</button>
          </form>
        </section>
        <section className="billing-card">
          <h2>Create campaign coupon</h2>
          <form className="stack-form" onSubmit={(event) => void createCoupon(event)}>
            <input name="label" placeholder="Campaign label" required />
            <input name="code" placeholder="Optional custom code" />
            <input name="amount" type="number" min="0.000001" step="0.000001" placeholder="USD credit" required />
            <input name="cap" type="number" min="1" placeholder="Redemption cap" required />
            <input name="expires" type="datetime-local" />
            <button type="submit">Create coupon</button>
          </form>
        </section>
      </div>
      <section className="billing-card">
        <h2>Coupon campaigns</h2>
        <div className="coupon-list">
          {coupons.map((coupon) => (
            <div key={coupon.id}>
              <strong>{coupon.label}</strong> <code>{coupon.code_hint}</code>
              <span>{formatUsd(coupon.amount_microusd)} · {coupon.redemption_count}/{coupon.max_redemptions}</span>
              {coupon.active && <button type="button" onClick={() => void deactivate(coupon.id)}>Deactivate</button>}
            </div>
          ))}
        </div>
      </section>
      <LedgerTable items={ledger} showUser />
    </BillingShell>
  );
}

function BillingShell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <main className="billing-page">
      <header className="billing-header">
        <div><a href="/">← Workspace</a><h1>{title}</h1></div>
        <UserButton />
      </header>
      {children}
    </main>
  );
}

function LedgerTable({ items, showUser }: { items: LedgerEvent[]; showUser: boolean }) {
  return (
    <section className="billing-card">
      <h2>Immutable ledger</h2>
      <div className="ledger-wrap"><table className="ledger-table">
        <thead><tr>{showUser && <th>User</th>}<th>When</th><th>Event</th><th>Attribution</th><th>Amount</th></tr></thead>
        <tbody>{items.map((item) => <tr key={item.id}>
          {showUser && <td><code>{item.user_id}</code></td>}
          <td>{new Date(item.created_at).toLocaleString()}</td>
          <td>{item.description ?? item.event_type}</td>
          <td><code>{item.provider_request_id ?? item.agent_span_id ?? "manual"}</code></td>
          <td className={item.amount_microusd > 0 ? "credit" : "debit"}>{formatSignedUsd(item.amount_microusd)}</td>
        </tr>)}</tbody>
      </table></div>
    </section>
  );
}

function formatUsd(value: number): string {
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 6 }).format(value / 1_000_000);
}

function formatSignedUsd(value: number): string {
  return `${value > 0 ? "+" : "−"}${formatUsd(Math.abs(value))}`;
}

async function errorDetail(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null) as { detail?: string } | null;
  return payload?.detail ?? `Request failed (${response.status})`;
}
