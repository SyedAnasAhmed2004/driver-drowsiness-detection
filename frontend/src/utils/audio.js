let audioCtx = null;

function ctx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  // Resume if suspended (browser autoplay policy)
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

function beep({ frequency, duration, volume, startOffset = 0 }) {
  const c = ctx();
  const osc  = c.createOscillator();
  const gain = c.createGain();

  osc.connect(gain);
  gain.connect(c.destination);

  osc.type = 'sine';
  osc.frequency.value = frequency;

  const t0 = c.currentTime + startOffset;
  gain.gain.setValueAtTime(0, t0);
  gain.gain.linearRampToValueAtTime(volume, t0 + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t0 + duration);

  osc.start(t0);
  osc.stop(t0 + duration + 0.01);
}

// Three rapid high-pitched beeps — for DANGER
export function playDangerAlert() {
  try {
    [0, 0.22, 0.44].forEach((offset) =>
      beep({ frequency: 1100, duration: 0.18, volume: 0.5, startOffset: offset })
    );
  } catch {
    // ignore if audio is blocked
  }
}

// Single softer beep — for MILD RISK
export function playMildAlert() {
  try {
    beep({ frequency: 660, duration: 0.25, volume: 0.3 });
  } catch {
    // ignore
  }
}
