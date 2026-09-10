(async function () {
  "use strict";

  const STORAGE_KEY = "anniversary-storybook:unlocked";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const els = {
    coverName: document.getElementById("cover-name"),
    beginBtn: document.getElementById("begin-btn"),
    chapters: document.getElementById("chapters"),
    rail: document.getElementById("progress-rail"),
    ambient: document.getElementById("ambient"),
    finale: document.getElementById("finale"),
    finaleLocked: document.getElementById("finale-locked"),
    finaleUnlocked: document.getElementById("finale-unlocked"),
    finaleLockedHeadline: document.getElementById("finale-locked-headline"),
    finaleLockedSub: document.getElementById("finale-locked-sub"),
    letterPhoto: document.getElementById("letter-photo"),
    letterTitle: document.getElementById("letter-title"),
    letterBody: document.getElementById("letter-body"),
    letterSignature: document.getElementById("letter-signature"),
    cdDays: document.getElementById("cd-days"),
    cdHours: document.getElementById("cd-hours"),
    cdMins: document.getElementById("cd-mins"),
    cdSecs: document.getElementById("cd-secs"),
  };

  let data;
  try {
    const res = await fetch("content/chapters.json");
    data = await res.json();
  } catch (err) {
    els.coverName.textContent = "Couldn't load the story";
    console.error("Failed to load content/chapters.json — are you running this via a local server?", err);
    return;
  }

  // ── Cover ──
  els.coverName.textContent = data.recipient || "—";

  // ── Chapters ──
  const sections = []; // { id, el }

  data.chapters.forEach((chapter) => {
    const section = document.createElement("section");
    section.className = "chapter";
    section.id = chapter.id;
    section.dataset.accent = chapter.accent || "";

    const photosHtml = (chapter.photos || [])
      .map((p) => `<img src="${escapeAttr(p.src)}" alt="${escapeAttr(p.alt || "")}" loading="lazy" />`)
      .join("");

    section.innerHTML = `
      <p class="chapter-date">${escapeHtml(chapter.date || "")}</p>
      <h2 class="chapter-title">${escapeHtml(chapter.title || "")}</h2>
      <p class="chapter-body">${escapeHtml(chapter.body || "")}</p>
      ${photosHtml ? `<div class="chapter-photos">${photosHtml}</div>` : ""}
    `;

    els.chapters.appendChild(section);
    sections.push({ id: chapter.id, el: section });
  });

  sections.push({ id: "finale", el: els.finale });

  // ── Progress rail ──
  sections.forEach(({ id }, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-label", `Go to chapter ${i + 1}`);
    btn.innerHTML = "<span></span>";
    btn.addEventListener("click", () => {
      document.getElementById(id).scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
    });
    els.rail.appendChild(btn);
  });
  const railButtons = Array.from(els.rail.children);

  function updateRail(activeId) {
    railButtons.forEach((btn, i) => {
      btn.setAttribute("aria-current", sections[i].id === activeId ? "true" : "false");
    });
  }

  // ── Reveal-on-scroll (IntersectionObserver, not scroll-jacking — see PRD §7.2) ──
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          updateRail(entry.target.id);
        }
      });
    },
    { threshold: 0.4 }
  );
  sections.forEach(({ el }) => {
    if (el.classList.contains("chapter")) observer.observe(el);
  });

  els.beginBtn.addEventListener("click", () => {
    const first = sections[0];
    if (first) document.getElementById(first.id).scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  });

  // ── Ambient background (subtle, tone-appropriate — not the Valentine hearts) ──
  if (!reduceMotion) {
    const symbols = ["·", "✦", "○"];
    function spawnAmbient() {
      const s = document.createElement("span");
      s.textContent = symbols[Math.floor(Math.random() * symbols.length)];
      s.style.left = Math.random() * 100 + "%";
      s.style.animationDuration = 10 + Math.random() * 10 + "s";
      s.style.animationDelay = Math.random() * 2 + "s";
      els.ambient.appendChild(s);
      setTimeout(() => s.remove(), 22000);
    }
    setInterval(spawnAmbient, 1200);
  }

  // ── Finale: countdown / unlock ──
  const unlockAt = new Date(data.unlockAt).getTime();
  els.finaleLockedHeadline.textContent = data.finale.lockedHeadline;
  els.finaleLockedSub.textContent = data.finale.lockedSubline;

  function renderCountdown() {
    const now = Date.now();
    const remaining = unlockAt - now;

    if (remaining <= 0) {
      unlockFinale();
      return;
    }

    const days = Math.floor(remaining / 86400000);
    const hours = Math.floor((remaining % 86400000) / 3600000);
    const mins = Math.floor((remaining % 3600000) / 60000);
    const secs = Math.floor((remaining % 60000) / 1000);

    els.cdDays.textContent = days;
    els.cdHours.textContent = String(hours).padStart(2, "0");
    els.cdMins.textContent = String(mins).padStart(2, "0");
    els.cdSecs.textContent = String(secs).padStart(2, "0");
  }

  function unlockFinale() {
    clearInterval(countdownTimer);
    els.finale.dataset.unlocked = "true";
    els.finale.classList.add("is-unlocked");
    els.finaleLocked.style.display = "none";
    els.finaleUnlocked.style.display = "flex";
    els.finaleUnlocked.style.flexDirection = "column";
    els.finaleUnlocked.style.alignItems = "center";
    els.finaleUnlocked.style.gap = "20px";

    els.letterPhoto.src = data.finale.photo?.src || "";
    els.letterPhoto.alt = data.finale.photo?.alt || "";
    els.letterTitle.textContent = data.finale.letterTitle;
    els.letterBody.textContent = data.finale.letterBody;
    els.letterSignature.textContent = data.finale.signature;

    const alreadyCelebrated = localStorage.getItem(STORAGE_KEY);
    if (!alreadyCelebrated) {
      localStorage.setItem(STORAGE_KEY, "1");
      if (!reduceMotion) startConfetti();
    }
  }

  let countdownTimer;
  if (Date.now() >= unlockAt) {
    unlockFinale();
  } else {
    renderCountdown();
    countdownTimer = setInterval(renderCountdown, 1000);
  }

  // ── Confetti (adapted from the original Valentine page's celebration effect) ──
  function startConfetti() {
    const canvas = document.getElementById("confetti-canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const colors = ["#b5654a", "#6c4a63", "#4c6b7a", "#5c7a52", "#7a6a8a", "#faf8f4"];
    const pieces = [];
    for (let i = 0; i < 120; i++) {
      pieces.push({
        x: Math.random() * canvas.width,
        y: -20 - Math.random() * canvas.height,
        w: 4 + Math.random() * 6,
        h: 4 + Math.random() * 6,
        color: colors[Math.floor(Math.random() * colors.length)],
        vx: Math.random() * 3 - 1.5,
        vy: 2 + Math.random() * 3,
        rotation: Math.random() * 360,
        rotSpeed: Math.random() * 8 - 4,
        opacity: 0.7 + Math.random() * 0.3,
      });
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let active = false;
      for (const p of pieces) {
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.rotSpeed;
        p.vy += 0.04;
        if (p.y < canvas.height + 40) active = true;

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate((p.rotation * Math.PI) / 180);
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        ctx.restore();
      }
      if (active) requestAnimationFrame(animate);
    }
    animate();

    window.addEventListener("resize", () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }, { once: true });
  }

  // ── Utilities ──
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
  function escapeAttr(str) {
    return escapeHtml(str).replace(/"/g, "&quot;");
  }
})();
