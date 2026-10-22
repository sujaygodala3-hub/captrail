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
  const logo = '<a class="logo" href="index.html"><svg><use href="#mark"/></svg>Captrail</a>';

  document.body.insertAdjacentHTML('afterbegin',
    `<svg width="0" height="0" style="position:absolute" aria-hidden="true">
      <symbol id="mark" viewBox="0 0 24 24"><path d="M5 22V10a3.5 3.5 0 0 1 7 0v12M12 22V6a4 4 0 0 1 8 0v16" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/></symbol>
    </svg>`);

  const headerSlot = document.getElementById('site-header');
  if (headerSlot) headerSlot.outerHTML = `
    <header class="header" id="header">
      <nav class="nav" aria-label="Main">
        ${logo}
        <div class="nav-links">${links}</div>
        <a class="btn btn-red" href="${CONTACT}">Contact us</a>
        <button class="menu-btn" id="menuBtn" aria-label="Open menu" aria-expanded="false">
          <svg width="20" height="20" viewBox="0 0 20 20"><path d="M3 7h14M3 13h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
        </button>
      </nav>
    </header>
    <div class="mobile-menu" id="mobileMenu"><a href="index.html">Home</a>${links}<a href="${CONTACT}">Contact us</a></div>`;

  const footerSlot = document.getElementById('site-footer');
  if (footerSlot) footerSlot.outerHTML = `
    <footer>
      <div class="wrap">
        <div class="foot-top">${logo}<nav><a href="index.html">Home</a>${links}<a href="${CONTACT}">Contact</a></nav></div>
        <div class="foot-bot"><span>© 2026 Captrail</span><span>Research prototype. Not a medical device and not for diagnosis.</span></div>
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

  /* reveal on scroll */
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .15 });
  document.querySelectorAll('.rv').forEach(el => io.observe(el));
})();
