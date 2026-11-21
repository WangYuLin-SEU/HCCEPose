/* Local documentation interactions; no external services.
   本地文档交互，不依赖外部服务。 */
(function () {
  'use strict';

  var root = document.documentElement;
  var languageButtons = document.querySelectorAll('.language-switch button[data-language]');
  var searchInput = document.getElementById('site-search');
  var searchPanel = document.getElementById('search-results');
  var searchList = document.getElementById('search-results-list');
  var searchStatus = document.getElementById('search-status');
  var searchIndex = null;
  var activeResult = -1;
  var searchRequest = 0;
  var docsBanner = document.querySelector('.docs-banner');
  var bannerCopy = document.querySelector('.docs-banner-copy');
  var siteHeader = document.querySelector('.site-header');
  var bannerScrollRange = 1;
  var bannerPeek = 120;
  var previousBannerProgress = -1;
  var bannerElements = Array.from(bannerCopy.querySelectorAll(':scope > .banner-kicker, :scope > .banner-title'));
  var bannerDescription = bannerCopy.querySelector(':scope > .banner-description');
  var bannerLink = bannerCopy.querySelector(':scope > a');
  var bannerGeometry = [];
  var arrivalProgress = Number(root.dataset.bannerRestore || 0);
  var navigation = performance.getEntriesByType('navigation')[0];
  var navigationType = navigation ? navigation.type : 'navigate';
  var pageIsShown = false;

  // An invisible compact layout supplies destinations; only the original text is ever displayed.
  // 隐藏的紧凑排版只提供终点坐标；始终显示原来的一套文字，不再淡出后切换副本。
  var retainedCopy = document.createElement('div');
  retainedCopy.className = 'docs-banner-retained';
  retainedCopy.setAttribute('aria-hidden', 'true');
  bannerElements.forEach(function (paragraph) {
    retainedCopy.appendChild(paragraph.cloneNode(true));
  });
  docsBanner.appendChild(retainedCopy);

  // Measure only at layout changes; transforms never change the banner or canvas dimensions.
  // 仅在布局变化时测量；文字变换不改变横幅占位及画布尺寸。
  function measureBanner() {
    bannerElements.forEach(function (paragraph) { paragraph.style.transform = ''; });
    var styles = getComputedStyle(root);
    bannerPeek = parseFloat(styles.getPropertyValue('--banner-peek'));
    var bannerGap = parseFloat(styles.getPropertyValue('--banner-gap'));
    var bannerHeight = docsBanner.getBoundingClientRect().height;
    root.style.setProperty('--banner-height', bannerHeight + 'px');
    bannerScrollRange = Math.max(1, bannerHeight + bannerGap - bannerPeek);
    bannerGeometry = bannerElements.map(function (paragraph, index) {
      var destination = retainedCopy.children[index];
      var start = paragraph.getBoundingClientRect();
      var end = destination.getBoundingClientRect();
      // Retained identity lines use the same typography, so only their position changes.
      // 保留区两行标识沿用原字号与行高，滚动只改变位置，不缩放文字或绕字动效。
      return {dx: end.left - start.left, dy: end.top - start.top};
    });
    previousBannerProgress = -1;
    updateBannerCopy();
  }

  function smoothBannerProgress(value) {
    var bounded = Math.max(0, Math.min(1, value));
    return bounded * bounded * (3 - 2 * bounded);
  }

  function updateBannerCopy() {
    var progress = Math.max(0, Math.min(1, window.scrollY / bannerScrollRange));
    if (progress === previousBannerProgress) return;
    previousBannerProgress = progress;
    var eased = smoothBannerProgress(progress);
    bannerElements.forEach(function (paragraph, index) {
      var geometry = bannerGeometry[index];
      var x = geometry.dx * eased;
      // Interpolate in viewport space, then compensate for the masthead's upward scroll.
      // 先在视口中插值，再补偿横幅自身的上移，避免文字先上冲、再掉回保留区域。
      var y = progress * bannerScrollRange + (geometry.dy - bannerScrollRange) * eased;
      var translation = 'translate3d(' + x.toFixed(3) + 'px,' + y.toFixed(3) + 'px,0)';
      paragraph.style.transform = translation;
      // Keep the fading details below the full-size title throughout the transition.
      // 淡出中的下方文案随标题平移，始终保持间距，避免与未缩小的字形相撞。
      if (index === 1) {
        bannerDescription.style.transform = translation;
        bannerLink.style.transform = translation;
      }
    });

    // The description and project link leave together; hidden copy also leaves the
    // accessibility tree and keyboard focus without changing the banner's layout.
    // 简介与项目入口同步淡出；消失后同时退出读屏和键盘焦点，不改变横幅占位。
    var detailFade = smoothBannerProgress(progress / 0.32);
    var detailOpacity = (1 - detailFade).toFixed(3);
    var detailsHidden = detailOpacity === '0.000';
    [bannerDescription, bannerLink].forEach(function (element) {
      element.style.opacity = detailOpacity;
      element.style.visibility = detailsHidden ? 'hidden' : '';
      element.inert = detailsHidden;
      element.setAttribute('aria-hidden', String(detailsHidden));
      if (detailsHidden && element.contains(document.activeElement)) document.activeElement.blur();
    });
  }
  measureBanner();
  if ('ResizeObserver' in window) {
    var bannerSizeObserver = new ResizeObserver(measureBanner);
    bannerSizeObserver.observe(docsBanner);
  }

  function language() { return root.dataset.lang === 'zh' ? 'zh' : 'en'; }
  function label(zh, en) { return language() === 'zh' ? zh : en; }

  function closeSearch() {
    searchRequest += 1;
    searchPanel.hidden = true;
    searchInput.setAttribute('aria-expanded', 'false');
    searchInput.removeAttribute('aria-activedescendant');
    activeResult = -1;
  }

  function setActiveResult(index) {
    var results = searchList.querySelectorAll('a');
    if (!results.length) return;
    activeResult = (index + results.length) % results.length;
    results.forEach(function (result, position) {
      result.setAttribute('aria-selected', String(position === activeResult));
    });
    searchInput.setAttribute('aria-activedescendant', results[activeResult].id);
    results[activeResult].scrollIntoView({block: 'nearest'});
  }

  // Fit results below the real input position, including an expanded banner or keyboard.
  // 按搜索框的实际位置计算剩余高度，兼顾展开的横幅和手机软键盘。
  function sizeSearchResults() {
    if (searchPanel.hidden) return;
    var viewport = window.visualViewport;
    var visibleBottom = viewport ? viewport.offsetTop + viewport.height : window.innerHeight;
    var available = visibleBottom - searchInput.getBoundingClientRect().bottom - 20;
    searchPanel.style.setProperty('--search-room', Math.max(0, available) + 'px');
  }

  async function showSearch() {
    var query = searchInput.value.trim().toLocaleLowerCase();
    var request = ++searchRequest;
    if (!query) { closeSearch(); return; }
    searchPanel.hidden = false;
    sizeSearchResults();
    searchInput.setAttribute('aria-expanded', 'true');
    searchStatus.textContent = label('正在搜索…', 'Searching…');
    searchList.replaceChildren();
    activeResult = -1;
    try {
      if (!searchIndex) {
        searchIndex = fetch('search-index.json?v=public-postprocess-20261121', {cache: 'no-cache'}).then(function (response) {
          if (!response.ok) throw new Error('Search index: HTTP ' + response.status);
          return response.json();
        });
      }
      var entries = await searchIndex;
      if (request !== searchRequest) return;
      var terms = query.split(/\s+/);
      var matches = entries.filter(function (entry) { return entry.lang === language(); }).map(function (entry) {
        var title = entry.title.toLocaleLowerCase();
        var text = entry.text.toLocaleLowerCase();
        var score = terms.every(function (term) { return text.includes(term) || title.includes(term); }) ? 1 : 0;
        if (score) terms.forEach(function (term) { if (title.includes(term)) score += 5; });
        return {entry: entry, score: score};
      }).filter(function (match) { return match.score > 0; }).sort(function (a, b) { return b.score - a.score; });
      searchStatus.textContent = matches.length ? label('找到 ' + matches.length + ' 篇文档', matches.length === 1 ? '1 matching page' : matches.length + ' matching pages') : label('没有匹配内容，试试函数名或关键词。', 'No matches. Try a function name or keyword.');
      matches.slice(0, 8).forEach(function (match, index) {
        var item = document.createElement('li');
        item.setAttribute('role', 'presentation');
        var link = document.createElement('a');
        link.href = match.entry.url;
        link.id = 'search-result-' + index;
        link.setAttribute('role', 'option');
        link.setAttribute('aria-selected', 'false');
        var title = document.createElement('strong');
        title.textContent = match.entry.title;
        var excerpt = document.createElement('span');
        var position = match.entry.text.toLocaleLowerCase().indexOf(terms[0]);
        var start = Math.max(0, position - 35);
        excerpt.textContent = (start ? '…' : '') + match.entry.text.slice(start, start + 150) + '…';
        link.append(title, excerpt);
        item.append(link);
        searchList.append(item);
      });
    } catch (error) {
      if (request !== searchRequest) return;
      searchIndex = null;
      searchStatus.textContent = label('搜索暂时不可用，请刷新页面后重试。', 'Search is temporarily unavailable. Refresh the page and try again.');
    }
  }

  function applyLanguage(nextLanguage, updateAddress) {
    var previousLanguage = language();
    var switchingLanguage = updateAddress && previousLanguage !== nextLanguage;
    var previousScrollY = window.scrollY;
    var oldHash = '';
    try { oldHash = decodeURIComponent(location.hash.slice(1)); } catch (error) {}
    var hashLanguage = oldHash.startsWith('en-') ? 'en' : oldHash.startsWith('zh-') ? 'zh' : previousLanguage;

    // Capture the visible section before hiding it; an old URL hash may be far away.
    // 隐藏当前语言前记录阅读章节；URL 中的旧锚点可能早已离开当前视口。
    var oldSection = document.querySelector('.doc-language.' + previousLanguage);
    var newSection = document.querySelector('.doc-language.' + nextLanguage);
    var oldHeadings = Array.from(oldSection.querySelectorAll('h1,h2,h3,h4'));
    var newHeadings = Array.from(newSection.querySelectorAll('h1,h2,h3,h4'));
    var sameStructure = oldHeadings.length === newHeadings.length && oldHeadings.every(function (heading, index) { return heading.tagName === newHeadings[index].tagName; });
    var readingIndex = -1;
    var sectionProgress = 0;
    if (switchingLanguage && sameStructure && oldHeadings.length) {
      var readingOffset = parseFloat(getComputedStyle(oldHeadings[0]).scrollMarginTop) + parseFloat(getComputedStyle(root).scrollPaddingTop);
      oldHeadings.forEach(function (heading, index) {
        if (Math.round(heading.getBoundingClientRect().top) <= Math.round(readingOffset)) readingIndex = index;
      });
      if (readingIndex >= 0) {
        var sectionStart = oldHeadings[readingIndex].getBoundingClientRect().top + previousScrollY;
        var sectionEnd = readingIndex + 1 < oldHeadings.length ? oldHeadings[readingIndex + 1].getBoundingClientRect().top + previousScrollY : oldSection.getBoundingClientRect().bottom + previousScrollY;
        if (sectionEnd > sectionStart) sectionProgress = Math.max(0, Math.min(1, (previousScrollY + readingOffset - sectionStart) / (sectionEnd - sectionStart)));
      }
    }

    root.dataset.lang = nextLanguage;
    root.lang = nextLanguage === 'zh' ? 'zh-CN' : 'en';
    document.title = document.body.dataset[nextLanguage === 'en' ? 'titleEn' : 'titleZh'] + ' · HCCEPose Docs';
    try { localStorage.setItem('hccepose-language', nextLanguage); } catch (error) {}
    languageButtons.forEach(function (button) { button.setAttribute('aria-pressed', String(button.dataset.language === nextLanguage)); });
    document.querySelectorAll('[data-label-zh]').forEach(function (element) {
      element.setAttribute('aria-label', label(element.dataset.labelZh, element.dataset.labelEn));
    });
    searchInput.placeholder = searchInput.dataset[nextLanguage === 'en' ? 'placeholderEn' : 'placeholderZh'];
    document.querySelectorAll('a[href]').forEach(function (link) {
      var reference = link.getAttribute('href');
      if (reference.startsWith('#')) return;
      var target = new URL(reference, location.href);
      // The shared project directory stays reachable from standalone local previews.
      // 独立本地预览也能返回共用项目目录，并保留当前语言。
      if (link.hasAttribute('data-project-directory')) {
        target.searchParams.set('lang', nextLanguage);
        link.setAttribute('href', target.href);
        return;
      }
      if (target.origin === location.origin && target.pathname.endsWith('.html')) {
        target.searchParams.set('lang', nextLanguage);
        // Keep relative addresses usable under any hosting prefix.
        // 保留相对地址，使网站可放在任意托管子路径下。
        link.setAttribute('href', reference.split(/[?#]/)[0] + target.search + target.hash);
      }
    });
    document.querySelectorAll('.copy-code').forEach(function (button) {
      button.textContent = label('复制', 'Copy');
      button.setAttribute('aria-label', label('复制代码', 'Copy code'));
    });
    document.querySelectorAll('.doc-language').forEach(function (section) {
      section.setAttribute('aria-hidden', String(!section.classList.contains(nextLanguage)));
    });
    // Initial cross-language deep links still resolve to their translated heading.
    // 首次打开跨语言深链接时，仍定位到结构对应的译文标题。
    var translatedHeading = null;
    if (!updateAddress && oldHash && hashLanguage !== nextLanguage) {
      var linkedHeadings = Array.from(document.querySelectorAll('.doc-language.' + hashLanguage + ' :is(h1,h2,h3,h4)'));
      var matchingLinkStructure = linkedHeadings.length === newHeadings.length && linkedHeadings.every(function (heading, index) { return heading.tagName === newHeadings[index].tagName; });
      var linkedIndex = linkedHeadings.findIndex(function (heading) { return heading.id === oldHash; });
      if (matchingLinkStructure && linkedIndex >= 0) translatedHeading = newHeadings[linkedIndex];
    }
    if (updateAddress || translatedHeading) {
      var address = new URL(location.href);
      address.searchParams.set('lang', nextLanguage);
      if (translatedHeading) address.hash = translatedHeading.id;
      else if (switchingLanguage && oldHash) address.hash = readingIndex >= 0 ? newHeadings[readingIndex].id : '';
      history.replaceState(history.state, '', address);
    }
    // Keep the same section and relative progress, allowing translations to differ in height.
    // 按对应章节和章内进度恢复阅读位置，允许译文高度不同，不将两种正文强行等高。
    requestAnimationFrame(function () {
      if (language() !== nextLanguage) return;
      if (translatedHeading) translatedHeading.scrollIntoView({block: 'start', behavior: 'instant'});
      else if (switchingLanguage) {
        var targetScrollY = previousScrollY;
        if (readingIndex >= 0) {
          var nextOffset = parseFloat(getComputedStyle(newHeadings[readingIndex]).scrollMarginTop) + parseFloat(getComputedStyle(root).scrollPaddingTop);
          var nextStart = newHeadings[readingIndex].getBoundingClientRect().top + window.scrollY;
          var nextEnd = readingIndex + 1 < newHeadings.length ? newHeadings[readingIndex + 1].getBoundingClientRect().top + window.scrollY : newSection.getBoundingClientRect().bottom + window.scrollY;
          targetScrollY = nextStart + (nextEnd - nextStart) * sectionProgress - nextOffset;
        }
        window.scrollTo({top: Math.max(0, targetScrollY), behavior: 'instant'});
      }
      updateReadingPosition();
    });
    if (searchInput.value.trim()) showSearch();
  }

  languageButtons.forEach(function (button) {
    button.addEventListener('click', function () { applyLanguage(button.dataset.language, true); });
  });
  applyLanguage(language(), false);

  searchInput.addEventListener('input', showSearch);
  searchInput.addEventListener('focus', function () { if (searchInput.value.trim()) showSearch(); });
  searchInput.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveResult(activeResult < 0 ? (event.key === 'ArrowDown' ? 0 : -1) : activeResult + (event.key === 'ArrowDown' ? 1 : -1));
    } else if (event.key === 'Enter' && !searchPanel.hidden && searchList.children.length) {
      event.preventDefault();
      searchList.querySelectorAll('a')[Math.max(activeResult, 0)].click();
    } else if (event.key === 'Escape') {
      closeSearch();
      searchInput.blur();
    }
  });
  document.addEventListener('click', function (event) {
    if (!event.target.closest('.search-box')) closeSearch();
  });
  document.addEventListener('keydown', function (event) {
    var typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || document.activeElement.isContentEditable;
    var shortcut = event.key === '/' && !typing && !event.ctrlKey && !event.metaKey && !event.altKey;
    var commandShortcut = event.key.toLowerCase() === 'k' && (event.ctrlKey || event.metaKey);
    if ((shortcut || commandShortcut) && !document.querySelector('dialog[open]')) {
      event.preventDefault();
      setMenu(false, false);
      searchInput.focus();
      searchInput.select();
    }
  });

  var menuButton = document.getElementById('menu-toggle');
  var sidebar = document.getElementById('sidebar');
  var mainContent = document.getElementById('main-content');
  var menuBackdrop = document.getElementById('menu-backdrop');
  var mobileViewport = matchMedia('(max-width: 800px)');

  // Keep keyboard navigation in the open drawer and release scroll lock on resize.
  // 打开目录时管理键盘焦点；切回桌面布局时释放滚动锁定。
  function setMenu(open, restoreFocus) {
    open = open && mobileViewport.matches;
    if (open) {
      var remainingBanner = Math.max(0, siteHeader.getBoundingClientRect().top - bannerPeek);
      if (remainingBanner > 0) window.scrollTo({top: window.scrollY + remainingBanner, behavior: 'instant'});
      updateBannerCopy();
      closeSearch();
    }
    menuButton.setAttribute('aria-expanded', String(open));
    document.body.classList.toggle('menu-open', open);
    menuBackdrop.hidden = !open;
    mainContent.inert = open;
    if (open) {
      // Focus the close action without scrolling a short drawer to its active page link.
      // 先聚焦关闭按钮，避免矮屏目录因定位当前页链接而把关闭入口滚出视口。
      sidebar.scrollTop = 0;
      document.getElementById('menu-close').focus({preventScroll: true});
    }
    else if (restoreFocus) menuButton.focus();
  }
  menuButton.addEventListener('click', function () {
    setMenu(menuButton.getAttribute('aria-expanded') !== 'true', false);
  });
  document.getElementById('menu-close').addEventListener('click', function () { setMenu(false, true); });
  menuBackdrop.addEventListener('click', function () { setMenu(false, true); });
  sidebar.addEventListener('click', function (event) { if (event.target.closest('a')) setMenu(false, false); });
  mobileViewport.addEventListener('change', function () { if (!mobileViewport.matches) setMenu(false, false); });
  document.addEventListener('keydown', function (event) {
    if (!document.body.classList.contains('menu-open')) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      setMenu(false, true);
    } else if (event.key === 'Tab') {
      var controls = Array.from(sidebar.querySelectorAll('a, button, summary')).filter(function (node) { return node.getClientRects().length; });
      var first = controls[0];
      var last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first || !sidebar.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !sidebar.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    }
  });

  var themeButton = document.getElementById('theme-toggle');
  themeButton.setAttribute('aria-pressed', String(root.dataset.theme === 'dark'));
  themeButton.addEventListener('click', function () {
    var theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = theme;
    themeButton.setAttribute('aria-pressed', String(theme === 'dark'));
    try { localStorage.setItem('hccepose-theme', theme); } catch (error) {}
  });

  document.querySelectorAll('pre').forEach(function (block) {
    var code = block.querySelector('code');
    if (!code) return;
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = label('复制', 'Copy');
    button.setAttribute('aria-label', label('复制代码', 'Copy code'));
    button.addEventListener('click', async function () {
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(code.textContent);
        } else {
          var temporary = document.createElement('textarea');
          temporary.value = code.textContent;
          temporary.style.position = 'fixed';
          temporary.style.opacity = '0';
          document.body.append(temporary);
          temporary.select();
          var copied = document.execCommand('copy');
          temporary.remove();
          button.focus();
          if (!copied) throw new Error('Clipboard unavailable');
        }
        button.textContent = label('已复制', 'Copied');
      } catch (error) {
        button.textContent = label('请选中代码复制', 'Select code to copy');
      }
      setTimeout(function () { button.textContent = label('复制', 'Copy'); }, 1800);
    });
    var toolbar = block.parentElement.querySelector('.code-toolbar');
    (toolbar || block).append(button);
  });

  var backToTop = document.getElementById('back-to-top');
  var scrollScheduled = false;
  function updateReadingPosition() {
    scrollScheduled = false;
    updateBannerCopy();
    // Save while this history entry is active; replacing state during pagehide disrupts Forward restoration.
    // 只在历史条目仍活动时保存；离页时 replaceState 会干扰浏览器前进后的阅读位置恢复。
    if (pageIsShown) {
      var progress = Math.max(0, Math.min(1, window.scrollY / bannerScrollRange));
      try {
        if (!history.state || history.state.hcceposeBannerProgress !== progress) {
          history.replaceState(Object.assign({}, history.state, {hcceposeBannerProgress: progress}), '', location.href);
        }
      } catch (error) {}
    }
    sizeSearchResults();
    backToTop.hidden = window.scrollY < 500;
    var headings = Array.from(document.querySelectorAll('.doc-language.' + language() + ' :is(h2,h3)'));
    var threshold = siteHeader.getBoundingClientRect().bottom + 45;
    var active = headings[0];
    headings.forEach(function (heading) { if (heading.getBoundingClientRect().top <= threshold) active = heading; });
    document.querySelectorAll('.section-outline a').forEach(function (link) {
      var current = active && decodeURIComponent(link.hash.slice(1)) === active.id;
      link.classList.toggle('is-active', Boolean(current));
      if (current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', function () {
    if (!scrollScheduled) { scrollScheduled = true; requestAnimationFrame(updateReadingPosition); }
  }, {passive: true});
  window.addEventListener('resize', function () {
    measureBanner();
    updateReadingPosition();
  });

  // Carry only the banner fraction into a new document, never the previous article's scroll depth.
  // 新文档只继承横幅收拢比例，不继承上一篇正文的滚动深度；锚点和新标签页保持原生行为。
  document.addEventListener('click', function (event) {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    var link = event.target.closest('a[href]');
    if (!link || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
    var target = new URL(link.href, location.href);
    var docsDirectory = new URL('.', location.href).pathname;
    if (target.origin !== location.origin || target.pathname === location.pathname || !target.pathname.endsWith('.html') || new URL('.', target).pathname !== docsDirectory) return;
    var progress = Math.max(0, Math.min(1, window.scrollY / bannerScrollRange));
    try {
      sessionStorage.setItem('hccepose-docs-banner-navigation', JSON.stringify({path: target.pathname, progress: progress, time: Date.now()}));
    } catch (error) {}
  });
  window.addEventListener('pagehide', function () { pageIsShown = false; });
  window.addEventListener('pageshow', function (event) {
    pageIsShown = true;
    requestAnimationFrame(function () {
      // Hash targets and restored history positions take precedence over a new-page banner offset.
      // 锚点及浏览器恢复的历史阅读位置优先；只有普通换页才滚到本页横幅的对应收拢位置。
      if (!event.persisted && navigationType === 'navigate' && arrivalProgress > 0 && !location.hash) {
        window.scrollTo({top: Math.ceil(bannerScrollRange * arrivalProgress), behavior: 'instant'});
      }
      measureBanner();
      updateReadingPosition();
      root.removeAttribute('data-banner-restore');
      arrivalProgress = 0;
    });
  });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', sizeSearchResults);
    window.visualViewport.addEventListener('scroll', sizeSearchResults);
  }
  backToTop.addEventListener('click', function () {
    window.scrollTo({top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
    document.querySelector('.brand').focus({preventScroll: true});
  });
  document.querySelector('.section-outline').addEventListener('click', function (event) {
    var link = event.target.closest('a');
    if (!link) return;
    this.open = false;
    requestAnimationFrame(function () {
      var target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (target) target.scrollIntoView({block: 'start'});
    });
  });

  // Enhance original-image links with a native modal; ordinary links remain the fallback.
  // 使用原生弹窗增强原图链接；无脚本或新标签操作仍使用普通图片地址。
  var imageViewer = document.getElementById('image-viewer');
  var imageOrigin = null;
  document.querySelectorAll('.image-original').forEach(function (link) {
    link.addEventListener('click', function (event) {
      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || !imageViewer.showModal) return;
      event.preventDefault();
      imageOrigin = link;
      var thumbnail = link.querySelector('img');
      var fullImage = document.getElementById('image-full');
      fullImage.src = link.href;
      fullImage.alt = thumbnail.alt;
      document.getElementById('image-original-link').href = link.href;
      document.getElementById('image-caption').textContent = thumbnail.alt;
      document.body.classList.add('image-open');
      imageViewer.showModal();
    });
  });
  document.getElementById('image-close').addEventListener('click', function () { imageViewer.close(); });
  imageViewer.addEventListener('click', function (event) { if (event.target === imageViewer) imageViewer.close(); });
  imageViewer.addEventListener('close', function () {
    document.body.classList.remove('image-open');
    document.getElementById('image-full').removeAttribute('src');
    if (imageOrigin) imageOrigin.focus({preventScroll: true});
  });
}());
