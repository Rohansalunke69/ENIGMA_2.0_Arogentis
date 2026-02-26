const BASE_URL = "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/api/v1/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

export async function analyzeEEG(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/api/v1/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Analysis failed" }));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}
