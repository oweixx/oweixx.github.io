export function parseSnapshot(value) {
  if (!value || value.timeZone !== 'Asia/Seoul' || !/^\d{4}-\d{2}-\d{2}$/.test(value.day)) throw new Error('Invalid visits response');
  for (const count of [value.total, value.today]) {
    if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid visit count');
  }
  for (const counts of [value.pages, value.days]) {
    if (!counts || typeof counts !== 'object' || Array.isArray(counts)) throw new Error('Invalid visit counts');
    for (const count of Object.values(counts)) {
      if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid visit count');
    }
  }
  return value;
}

export function recentDates(days, today) {
  return Array.from({ length: days }, (_, index) => {
    const date = new Date(`${today}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() - (days - index - 1));
    return date.toISOString().slice(0, 10);
  });
}
