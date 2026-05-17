export const CLASS_NAMES = [
  'c0: Safe driving',
  'c1: Texting (right hand)',
  'c2: Phone call (right hand)',
  'c3: Texting (left hand)',
  'c4: Phone call (left hand)',
  'c5: Radio / adjusting stereo',
  'c6: Drinking',
  'c7: Reaching behind',
  'c8: Hair / makeup',
  'c9: Talking to passenger',
];

// 0 = safe | 1 = mild | 2 = danger  (matches config.py SEVERITY)
export const SEVERITY = { 0:0, 1:2, 2:2, 3:2, 4:2, 5:1, 6:1, 7:2, 8:1, 9:1 };

export const SEV_COLOR = {
  0: '#22c55e',  // green
  1: '#f97316',  // orange
  2: '#ef4444',  // red
};

export const SEV_LABEL = {
  0: 'SAFE',
  1: 'MILD RISK',
  2: 'DANGER',
};

// Short axis labels for the bar chart
export const SHORT_NAMES = [
  'Safe', 'Txt-R', 'Ph-R', 'Txt-L', 'Ph-L',
  'Radio', 'Drink', 'Reach', 'Groom', 'Talk',
];

export const API_BASE = 'http://localhost:8000';
