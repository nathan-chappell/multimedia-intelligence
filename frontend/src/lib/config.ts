const optionalString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

export const config = {
  chatkitUrl: optionalString(import.meta.env.VITE_CHATKIT_API_URL) ?? "/chatkit",
  bearerToken:
    optionalString(import.meta.env.VITE_API_BEARER_TOKEN) ??
    "local-development-admin-token",
  domainKey:
    optionalString(import.meta.env.VITE_CHATKIT_API_DOMAIN_KEY) ??
    "domain_pk_localhost_dev",
};

export const authenticatedFetch: typeof fetch = (input, init = {}) => {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${config.bearerToken}`);
  return fetch(input, { ...init, headers });
};
