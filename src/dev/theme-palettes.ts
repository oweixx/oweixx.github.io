export const palettes = [
  { id: 'current', name: '현재 디자인', note: '적용된 Deep Teal', color: '#16766f', hover: '#105751', tint: '#f1f8f7', border: '#bedbd5' },
  { id: 'blue', name: 'Ocean Blue', note: '차분하고 선명한 블루', color: '#3563a8', hover: '#25497e', tint: '#f3f6fc', border: '#c5d3ea' },
  { id: 'teal', name: 'Deep Teal', note: '푸른빛이 도는 청록', color: '#16766f', hover: '#105751', tint: '#f1f8f7', border: '#bedbd5' },
  { id: 'forest', name: 'Forest Green', note: '차분한 짙은 초록', color: '#3f6b4f', hover: '#2c5039', tint: '#f4f8f3', border: '#cadcc9' },
  { id: 'plum', name: 'Muted Plum', note: '부드러운 자주빛', color: '#785476', hover: '#593e57', tint: '#faf5f9', border: '#dfcddd' },
  { id: 'terracotta', name: 'Terracotta', note: '따뜻한 적갈색', color: '#a55239', hover: '#7c3c28', tint: '#fcf5f1', border: '#e8cdbf' },
] as const;

export type Palette = (typeof palettes)[number];
