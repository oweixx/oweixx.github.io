export const badgeVariants = [
  { id: 'current', number: '00', name: '이전 배지', note: '기존 채움형 · 기본 굵기 400', css: '' },
  { id: 'medium', number: '01', name: 'Medium', note: '크기는 그대로, 글씨만 조금 진하게', css: 'font-weight: 500;' },
  { id: 'bold', number: '02', name: 'Bold', note: '글씨를 확실하게 강조 · 추천', css: 'font-weight: 700;' },
  { id: 'depth', number: '03', name: 'Soft shadow', note: '도톰한 글씨와 은은한 그림자 · 추천', css: 'font-weight: 600; box-shadow: 0 2px 7px rgb(24 52 91 / 24%), 0 0 0 1px rgb(255 255 255 / 65%);' },
  { id: 'outline', number: '04', name: 'White label', note: '흰 바탕에 파란 글씨와 테두리', css: 'font-weight: 600; background: #fff; color: var(--blue); border: 1px solid var(--blue); box-shadow: 0 2px 5px rgb(0 0 0 / 6%);' },
  { id: 'pill', number: '05', name: 'Rounded', note: '둥근 모서리로 부드럽게 강조', css: 'font-weight: 600; border-radius: 999px; padding: 4px 11px;' },
  { id: 'split', number: '06', name: 'Split year', note: '학회·저널과 연도를 두 칸으로 구분', css: 'display: inline-flex; align-items: stretch; padding: 0; overflow: hidden; border: 1px solid var(--blue); font-weight: 600;', childCss: '.badge-venue { padding: 4px 8px; } .badge-year { padding: 4px 7px; background: #fff; color: var(--blue); font-weight: 500; }' },
  { id: 'outline-bold', number: '07', name: 'White label + Bold', note: '선택한 스타일 · 흰 바탕과 굵은 글씨 700', css: 'font-weight: 700; background: #fff; color: var(--blue); border: 1px solid var(--blue); box-shadow: 0 2px 5px rgb(0 0 0 / 6%);' },
] as const;

export type BadgeVariant = (typeof badgeVariants)[number];
export const defaultBadgeVariant = badgeVariants.find(variant => variant.id === 'outline-bold')!;

export function badgeCss(selector: string, variant: BadgeVariant) {
  // Keep each demo independent of the badge currently used on the homepage.
  let css = `${selector} { display: inline-block; padding: 4px 9px; border: 0; border-radius: 3px; background: var(--blue); color: #fff; font-size: 12px; font-weight: 400; line-height: 1.3; box-shadow: none; overflow: visible; ${variant.css} }`;
  if ('childCss' in variant) css += variant.childCss.replace(/\.badge-(venue|year)/g, `${selector} .badge-$1`);
  return css;
}
