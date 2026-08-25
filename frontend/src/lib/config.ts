const optionalString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

type TokenGetter = () => Promise<string | null>;
let clerkTokenGetter: TokenGetter | undefined;

export const setClerkTokenGetter = (getter: TokenGetter | undefined): void => {
  clerkTokenGetter = getter;
};

export const config = {
  chatkitUrl: optionalString(import.meta.env.VITE_CHATKIT_API_URL) ?? "/chatkit",
  domainKey:
    optionalString(import.meta.env.VITE_CHATKIT_API_DOMAIN_KEY) ??
    "domain_pk_localhost_dev",
};

export const authenticatedFetch: typeof fetch = async (input, init = {}) => {
  const headers = new Headers(init.headers);
  const token = await clerkTokenGetter?.();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers });
};
