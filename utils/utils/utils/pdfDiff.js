const Diff = require("diff");
const { bboxUnion } = require("./helpers");

function indexWords(words) {
  // returns: tokens[], bboxes[] aligned by index
  const tokens = [];
  const boxes = [];
  for (const w of words) {
    tokens.push(w.text);
    boxes.push({ x: w.x, y: w.y, w: w.w, h: w.h });
  }
  return { tokens, boxes };
}

function buildPageDiff(oldPage, newPage) {
  const { tokens: a, boxes: aBoxes } = indexWords(oldPage.words);
  const { tokens: b, boxes: bBoxes } = indexWords(newPage.words);

  // Join tokens with space so the word-diff is stable
  const aStr = a.join(" ");
  const bStr = b.join(" ");

  const diff = Diff.diffWordsWithSpace(aStr, bStr);

  // Map ranges back to word indexes
  // We'll walk both token arrays using cursors
  let ai = 0, bi = 0;
  const changes = [];

  for (const part of diff) {
    const words = part.value.split(/\s+/).filter(Boolean);
    if (part.added) {
      // Words exist only in new → green
      const start = bi;
      const count = words.length;
      const bGroup = bBoxes.slice(start, start + count);
      if (bGroup.length) {
        const box = bGroup.reduce(bboxUnion);
        changes.push({
          type: "added",
          newText: words.join(" "),
          box,
          pageScale: { width: newPage.width, height: newPage.height }
        });
      }
      bi += count;
    } else if (part.removed) {
      // Words exist only in old → removed (show in summary; also mark red box where it was)
      const start = ai;
      const count = words.length;
      const aGroup = aBoxes.slice(start, start + count);
      if (aGroup.length) {
        const box = aGroup.reduce(bboxUnion);
        changes.push({
          type: "removed",
          oldText: words.join(" "),
          box,
          pageScale: { width: oldPage.width, height: oldPage.height }
        });
      }
      ai += count;
    } else {
      // Unchanged section
      ai += words.length;
      bi += words.length;
    }
  }

  // Derive "changed" by pairing remove+add that overlap nearby horizontally.
  // Simple heuristic: if a removed and an added box overlap > 20%, mark them as "changed".
  const added = changes.filter(c => c.type === "added");
  const removed = changes.filter(c => c.type === "removed");
  const toRemove = new Set();
  const changed = [];

  const iou = (b1, b2) => {
    const x1 = Math.max(b1.x, b2.x);
    const y1 = Math.max(b1.y, b2.y);
    const x2 = Math.min(b1.x + b1.w, b2.x + b2.w);
    const y2 = Math.min(b1.y + b1.h, b2.y + b2.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = b1.w * b1.h, area2 = b2.w * b2.h;
    const union = area1 + area2 - inter;
    return union ? inter / union : 0;
  };

  for (let r = 0; r < removed.length; r++) {
    for (let a = 0; a < added.length; a++) {
      if (toRemove.has(`r${r}`) || toRemove.has(`a${a}`)) continue;
      if (iou(removed[r].box, added[a].box) > 0.2) {
        changed.push({
          type: "changed",
          oldText: removed[r].oldText,
          newText: added[a].newText,
          box: bboxUnion(removed[r].box, added[a].box),
          pageScale: removed[r].pageScale // pages are same here
        });
        toRemove.add(`r${r}`);
        toRemove.add(`a${a}`);
      }
    }
  }

  const filtered = [
    ...changed,
    ...removed.filter((_, idx) => !toRemove.has(`r${idx}`)),
    ...added.filter((_, idx) => !toRemove.has(`a${idx}`))
  ];

  return filtered;
}

function diffDocuments(oldDoc, newDoc) {
  const pageCount = Math.max(oldDoc.pageCount, newDoc.pageCount);
  const pages = [];
  let addCount = 0, remCount = 0, chgCount = 0;

  for (let i = 0; i < pageCount; i++) {
    const oldPage = oldDoc.pages[i] || { width: newDoc.pages[i]?.width || 595, height: newDoc.pages[i]?.height || 842, words: [] };
    const newPage = newDoc.pages[i] || { width: oldDoc.pages[i]?.width || 595, height: oldDoc.pages[i]?.height || 842, words: [] };

    const diffs = buildPageDiff(oldPage, newPage);
    const pageChanges = {
      pageIndex: i,
      width: newPage.width,
      height: newPage.height,
      changes: diffs
    };

    diffs.forEach(d => {
      if (d.type === "added") addCount++;
      else if (d.type === "removed") remCount++;
      else if (d.type === "changed") chgCount++;
    });

    pages.push(pageChanges);
  }

  const summary = {
    totalPagesOld: oldDoc.pageCount,
    totalPagesNew: newDoc.pageCount,
    counts: { added: addCount, removed: remCount, changed: chgCount }
  };

  return { summary, pages };
}

module.exports = { diffDocuments };
