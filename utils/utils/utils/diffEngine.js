const { Diff } = require("diff");

// Round helpers to tolerate tiny shifts
const rxNum = /[-+]?(?:\d+([.,]\d+)?|\d*[.,]\d+)/;

function normText(t) {
  return (t || "").replace(/\s+/g, " ").trim();
}

function nearby(a, b, tol = 6) {
  return Math.abs(a - b) <= tol;
}

function keyFor(token) {
  // Position rounded to 5 units to reduce false changes when PDFs shift slightly
  const rx = Math.round(token.x / 5);
  const ry = Math.round(token.y / 5);
  return `${token.text.toLowerCase()}@${rx},${ry}`;
}

// Greedy alignment: match same word near the same spot => unchanged
function align(oldTokens, newTokens) {
  const oldBuckets = new Map();
  for (const t of oldTokens) {
    const k = keyFor(t);
    if (!oldBuckets.has(k)) oldBuckets.set(k, []);
    oldBuckets.get(k).push(t);
  }

  const unchanged = new Set();
  const added = [];
  for (const nt of newTokens) {
    const k = keyFor(nt);
    const list = oldBuckets.get(k);
    if (list && list.length) {
      // consume one
      list.pop();
      unchanged.add(nt);
    } else {
      added.push(nt);
    }
  }

  const removed = [];
  for (const [k, list] of oldBuckets.entries()) {
    for (const ot of list) removed.push(ot);
  }

  return { added, removed, unchanged };
}

// Pair likely value changes (number replaced) within same line neighborhood
function pairValueChanges(removed, added) {
  const pairs = [];
  const remainingAdds = [...added];

  for (const r of removed) {
    if (!rxNum.test(r.text)) continue;
    // find nearest numeric add on same page, close y & x
    let idx = -1;
    let best = null;
    for (let i = 0; i < remainingAdds.length; i++) {
      const a = remainingAdds[i];
      if (a.page !== r.page) continue;
      if (!rxNum.test(a.text)) continue;
      if (!nearby(a.y, r.y, 8)) continue;
      const dx = Math.abs(a.x - r.x);
      if (dx <= 80 && (!best || dx < best.dx)) best = { i, dx, a };
    }
    if (best) {
      pairs.push({ page: r.page, old: r, new: best.a });
      remainingAdds.splice(best.i, 1); // consume
    }
  }
  return { pairs, leftovers: remainingAdds };
}

function computeChanges(oldTokens, newTokens) {
  // per page
  const byPage = {};
  const pages = new Set([...oldTokens.map(t => t.page), ...newTokens.map(t => t.page)]);

  const allAdded = [];
  const allRemoved = [];
  const valuePairs = [];

  for (const p of pages) {
    const o = oldTokens.filter(t => t.page === p);
    const n = newTokens.filter(t => t.page === p);

    const { added, removed } = align(o, n);
    const { pairs, leftovers } = pairValueChanges(removed, added);

    valuePairs.push(...pairs);
    allAdded.push(...leftovers);
    allRemoved.push(...removed.filter(r => !pairs.find(pp => pp.old === r)));

    byPage[p] = {
      added: leftovers,
      removed: removed.filter(r => !pairs.find(pp => pp.old === r)),
      valuePairs: pairs
    };
  }

  return {
    added: allAdded,
    removed: allRemoved,
    valuePairs,
    byPage
  };
}

module.exports = { computeChanges };
