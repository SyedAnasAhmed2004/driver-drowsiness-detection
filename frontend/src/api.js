import { API_BASE } from './constants';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Backend unreachable');
  return res.json();
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/api/config`);
  return res.json();
}

export async function predictFrame(blob) {
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');
  const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return res.json();
}

export async function setThreshold(threshold) {
  await fetch(`${API_BASE}/api/threshold/${threshold}`, { method: 'POST' });
}
