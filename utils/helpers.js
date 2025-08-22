// Simple helpers used across utils
function tokenizeLine(text) {
  // split by spaces but keep punctuation glued to tokens
  return text.trim().split(/\s+/).filter(Boolean);
}

function bboxUnion(b1, b2) {
  return {
    x: Math.min(b1.x, b2.x),
    y: Math.min(b1.y, b2.y),
    w: Math.max(b1.x + b1.w, b2.x + b2.w) - Math.min(b1.x, b2.x),
    h: Math.max(b1.y + b1.h, b2.y + b2.h) - Math.min(b1.y, b2.y)
  };
}

module.exports = { tokenizeLine, bboxUnion };
