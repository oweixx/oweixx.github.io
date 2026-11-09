export const palettes = [
  { id: 'current', name: '현재 디자인', note: '적용된 KAIST Dark Blue', color: '#004191', hover: '#00316d', tint: '#f2f6fa', border: '#bfcee4' },
  // KAIST UI: Dark Blue RGB 0, 65, 145. Hover and surface tones are derived for this site.
  // https://www.kaist.ac.kr/en/html/kaist/011901.html
  { id: 'kaist', name: 'KAIST Dark Blue', note: 'KAIST 공식 로고 색상', color: '#004191', hover: '#00316d', tint: '#f2f6fa', border: '#bfcee4' },
  { id: 'pink', name: 'Dusty Pink · 15', note: '15번 핑크를 한 톤 밝게', color: '#ca83a1', hover: '#ad5c80', tint: '#fdf7fa', border: '#edd4e0' },
  { id: 'blue', name: 'Ocean Blue', note: '차분하고 선명한 블루', color: '#3563a8', hover: '#25497e', tint: '#f3f6fc', border: '#c5d3ea' },
  { id: 'teal', name: 'Deep Teal', note: '푸른빛이 도는 청록', color: '#16766f', hover: '#105751', tint: '#f1f8f7', border: '#bedbd5' },
  { id: 'forest', name: 'Forest Green', note: '차분한 짙은 초록', color: '#3f6b4f', hover: '#2c5039', tint: '#f4f8f3', border: '#cadcc9' },
  { id: 'plum', name: 'Muted Plum', note: '부드러운 자주빛', color: '#785476', hover: '#593e57', tint: '#faf5f9', border: '#dfcddd' },
  { id: 'terracotta', name: 'Terracotta', note: '따뜻한 적갈색', color: '#a55239', hover: '#7c3c28', tint: '#fcf5f1', border: '#e8cdbf' },
] as const;

export type Palette = (typeof palettes)[number];
export const defaultPalette = palettes.find(palette => palette.id === 'kaist')!;
