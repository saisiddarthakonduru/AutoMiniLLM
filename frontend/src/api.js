const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return res.json();
}

export const getHealth = () => request("/health");
export const getSystem = () => request("/system");
export const getModels = () => request("/models");
export const getHistory = () => request("/history");
export const compress = (payload) =>
  request("/compress", { method: "POST", body: JSON.stringify(payload) });
export const removeExperiment = (id) =>
  request(`/history/${id}`, { method: "DELETE" });
export const clearHistory = () =>
  request("/history", { method: "DELETE" });
export const compare = (expA, expB) =>
  request("/compare", { method: "POST", body: JSON.stringify({ exp_a: expA, exp_b: expB }) });
export const generateText = (payload) =>
  request("/generate", { method: "POST", body: JSON.stringify(payload) });
