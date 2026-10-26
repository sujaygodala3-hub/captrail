/* Shared header + footer for every Captrail page.
   To add a page to the header, add it to PAGES. */
(() => {
  const PAGES = [
    { href: 'build.html', label: 'Build guide' },
    { href: 'scan.html', label: 'Microscope' },
  ];
  const here = location.pathname.split('/').pop() || 'index.html';
  // Contact scrolls to the Get in touch section on the home page.
  const CONTACT = here === 'index.html' ? '#contact' : 'index.html#contact';
  const links = PAGES.map(p => `<a href="${p.href}"${p.href === here ? ' aria-current="page"' : ''}>${p.label}</a>`).join('');
  const REPO = 'https://github.com/sujaygodala3-hub/captrail';
  const gh = `<a class="gh-link" href="${REPO}" target="_blank" rel="noopener" aria-label="Captrail on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><use href="#gh"/></svg><span>GitHub</span></a>`;
  const logo = '<a class="logo" href="index.html" aria-label="Captrail home">'
    + '<img class="logo-c" src="images/brand/captrail-logo.svg" alt="Captrail" width="106" height="28">'
    + '<img class="logo-w" src="images/brand/captrail-logo-white.svg" alt="" aria-hidden="true" width="106" height="28"></a>';

  document.body.insertAdjacentHTML('afterbegin',
    `<svg width="0" height="0" style="position:absolute" aria-hidden="true">
      <symbol id="gh" viewBox="0 0 16 16"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></symbol>
    </svg>`);

  const headerSlot = document.getElementById('site-header');
  if (headerSlot) headerSlot.outerHTML = `
    <header class="header" id="header">
      <nav class="nav" aria-label="Main">
        ${logo}
        <div class="nav-links">${links}</div>
        ${gh}
        <a class="btn btn-red" href="${CONTACT}">Contact us</a>
        <button class="menu-btn" id="menuBtn" aria-label="Open menu" aria-expanded="false">
          <svg width="20" height="20" viewBox="0 0 20 20"><path d="M3 7h14M3 13h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
        </button>
      </nav>
    </header>
    <div class="mobile-menu" id="mobileMenu"><a href="index.html">Home</a>${links}<a href="${CONTACT}">Contact us</a><a href="${REPO}" target="_blank" rel="noopener">GitHub ↗</a></div>`;

  const footerSlot = document.getElementById('site-footer');
  if (footerSlot) footerSlot.outerHTML = `
    <footer>
      <div class="wrap">
        <div class="foot-top">${logo}<nav><a href="index.html">Home</a>${links}<a href="${CONTACT}">Contact</a><a href="${REPO}" target="_blank" rel="noopener">GitHub ↗</a></nav></div>
        <div class="foot-bot"><span>© 2026 Captrail · <a href="${REPO}" target="_blank" rel="noopener">Open source on GitHub</a></span><span>Research prototype. Not a medical device and not for diagnosis.</span></div>
      </div>
    </footer>`;

  /* header turns dark over dark sections */
  const header = document.getElementById('header');
  const darkSecs = [...document.querySelectorAll('[data-header="dark"]')];
  const headerTheme = () => header && header.classList.toggle('dark',
    darkSecs.some(s => { const r = s.getBoundingClientRect(); return r.top <= 40 && r.bottom >= 40; }));
  addEventListener('scroll', headerTheme, { passive: true });
  addEventListener('resize', headerTheme);
  headerTheme();

  /* mobile menu */
  const menuBtn = document.getElementById('menuBtn'), menu = document.getElementById('mobileMenu');
  if (menuBtn) {
    menuBtn.addEventListener('click', () => { const o = menu.classList.toggle('open'); menuBtn.setAttribute('aria-expanded', o); });
    menu.addEventListener('click', e => { if (e.target.tagName === 'A') menu.classList.remove('open'); });
  }

  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* clips: play only while on screen; with reduced motion, show controls instead */
  const clips = [...document.querySelectorAll('video.autoclip')];
  if (reduce) clips.forEach(v => { v.controls = true; v.preload = 'metadata'; });
  else if (clips.length) {
    const vo = new IntersectionObserver(es => es.forEach(e => {
      const v = e.target;
      if (e.isIntersecting) { if (v.preload === 'none') v.preload = 'auto'; v.play().catch(err => { if (err && err.name !== 'AbortError') v.controls = true; }); }
      else v.pause();
    }), { threshold: .35 });
    clips.forEach(v => vo.observe(v));
  }

  /* count-up numbers */
  const nums = [...document.querySelectorAll('[data-count]')];
  if (!reduce && nums.length) {
    const fmt = (el, x) => (el.dataset.pre || '') + x.toFixed(+el.dataset.dec || 0);
    const co = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return;
      co.unobserve(e.target);
      const el = e.target, end = parseFloat(el.dataset.count), t0 = performance.now(), dur = 1200;
      const step = now => { const k = Math.min((now - t0) / dur, 1); el.textContent = fmt(el, end * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    }), { threshold: .6 });
    nums.forEach(el => co.observe(el));
  }

  /* reveal on scroll */
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .15 });
  document.querySelectorAll('.rv').forEach(el => io.observe(el));
})();
