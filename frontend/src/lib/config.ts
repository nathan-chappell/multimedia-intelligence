const optionalString = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined;

const TOKEN_STORAGE_KEY = "api_bearer_token";
const AUTH_ERROR_STORAGE_KEY = "auth_error";

export const getBearerToken = (): string | undefined =>
  optionalString(window.localStorage.getItem(TOKEN_STORAGE_KEY)) ??
  optionalString(import.meta.env.VITE_API_BEARER_TOKEN);

export const storeBearerToken = (token: string): void => {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  window.sessionStorage.removeItem(AUTH_ERROR_STORAGE_KEY);
};

export const clearBearerToken = (): void => {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
};

export const getAuthError = (): string | undefined =>
  optionalString(window.sessionStorage.getItem(AUTH_ERROR_STORAGE_KEY));

export const markAuthFailure = (message: string): void => {
  window.sessionStorage.setItem(AUTH_ERROR_STORAGE_KEY, message);
};

export const config = {
  chatkitUrl: optionalString(import.meta.env.VITE_CHATKIT_API_URL) ?? "/chatkit",
  domainKey:
    optionalString(import.meta.env.VITE_CHATKIT_API_DOMAIN_KEY) ??
    "domain_pk_localhost_dev",
};

export const authenticatedFetch: typeof fetch = (input, init = {}) => {
  const headers = new Headers(init.headers);
  const token = getBearerToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(input, { ...init, headers }).then((response) => {
    if (response.status === 401 && window.location.pathname !== "/login") {
      markAuthFailure("Your session is invalid or has expired.");
      window.location.assign("/auth-error");
    }
    return response;
  });
};
