// Local landing interactions; no requests beyond explicitly viewed assets.
// 宣传首页交互；仅按用户选择加载演示资源，不依赖外部脚本。
(function () {
  'use strict';
  var root = document.documentElement;
  var languageButtons = Array.from(document.querySelectorAll('[data-language]'));
  var themeButton = document.getElementById('landing-theme-toggle');
  var menuButton = document.getElementById('landing-menu-toggle');
  var navigation = document.getElementById('landing-nav');
  var caseTabs = document.querySelector('.lp-case-tabs');
  var tabs = Array.from(document.querySelectorAll('.lp-case-tabs [role="tab"]'));
  var videoElement = document.getElementById('landing-video');
  var videoPlaying = false;
  var scenePreview = document.querySelector('[data-scene-viewer]');
  var sceneAlignButton = document.querySelector('[data-scene-align]');
  var sceneStateObserver = null;

  // Keep preview actions in the caption so they never cover a narrow 3D canvas.
  // 预览操作放在图注中，避免遮住窄屏三维画面；场景准备好后再开放复位。
  function watchScenePreview() {
    if (sceneStateObserver) sceneStateObserver.disconnect();
    var previewDocument = scenePreview.contentDocument;
    sceneAlignButton.disabled = true;
    if (!previewDocument) return;
    function updateSceneAvailability() {
      sceneAlignButton.disabled = previewDocument.documentElement.dataset.ready !== 'true';
    }
    sceneStateObserver = new MutationObserver(updateSceneAvailability);
    sceneStateObserver.observe(previewDocument.documentElement, {attributes: true, attributeFilter: ['data-ready']});
    updateSceneAvailability();
  }
  scenePreview.addEventListener('load', watchScenePreview);
  watchScenePreview();
  sceneAlignButton.addEventListener('click', function () {
    var previewDocument = scenePreview.contentDocument;
    if (previewDocument && previewDocument.documentElement.dataset.ready === 'true') {
      previewDocument.querySelector('[data-align-image]').click();
    }
  });

  // A local light follows a fine pointer; no animation loop runs while the pointer is idle.
  // 局部柔光跟随精细指针；指针静止时不持续循环，减弱动态偏好下不跟随。
  var ambientMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  document.querySelectorAll('[data-ambient-panel]').forEach(function (panel) {
    var ambientFrame = null;
    var pointerX = 0;
    var pointerY = 0;
    panel.addEventListener('pointermove', function (event) {
      if (ambientMotion.matches || !finePointer.matches) return;
      var bounds = panel.getBoundingClientRect();
      pointerX = event.clientX - bounds.left;
      pointerY = event.clientY - bounds.top;
      if (ambientFrame !== null) return;
      ambientFrame = window.requestAnimationFrame(function () {
        panel.style.setProperty('--pointer-x', pointerX + 'px');
        panel.style.setProperty('--pointer-y', pointerY + 'px');
        ambientFrame = null;
      });
    });
    panel.addEventListener('pointerleave', function () {
      if (ambientFrame !== null) window.cancelAnimationFrame(ambientFrame);
      ambientFrame = null;
    });
  });

  function updateMenuLabel() {
    var open = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-label', root.dataset.lang === 'en'
      ? (open ? 'Close navigation' : 'Open navigation')
      : (open ? '关闭导航' : '打开导航'));
  }

  // Retain explicit language in every local navigation, including docs links.
  // 所有站内跳转都携带当前语言，包括进入文档及返回主页。
  function applyLanguage(language, updateUrl) {
    root.dataset.lang = language;
    root.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.title = language === 'zh' ? 'HCCEPose · 已知物体的 6D 位姿估计' : 'HCCEPose · 6D pose estimation for known objects';
    document.querySelectorAll('[data-label-zh]').forEach(function (element) {
      element.setAttribute('aria-label', language === 'en' ? element.dataset.labelEn : element.dataset.labelZh);
    });
    document.querySelectorAll('img[data-alt-zh]').forEach(function (element) {
      element.alt = language === 'en' ? element.dataset.altEn : element.dataset.altZh;
    });
    document.querySelectorAll('[data-scene-viewer]').forEach(function (frame) {
      frame.title = language === 'en' ? frame.dataset.labelEn : frame.dataset.labelZh;
      var url = new URL(frame.getAttribute('src'), location.href);
      url.searchParams.set('lang', language);
      if (frame.src !== url.href) {
        sceneAlignButton.disabled = true;
        frame.src = url.href;
      }
    });
    updateMenuLabel();
    languageButtons.forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.language === language));
    });
    document.querySelectorAll('a[data-local-link]').forEach(function (link) {
      var originalHref = link.dataset.originalHref || link.getAttribute('href');
      link.dataset.originalHref = originalHref;
      var pathAndQuery = originalHref.split('#')[0];
      var path = pathAndQuery.split('?')[0];
      var query = new URLSearchParams(pathAndQuery.includes('?') ? pathAndQuery.slice(pathAndQuery.indexOf('?') + 1) : '');
      var hash = originalHref.includes('#') ? originalHref.slice(originalHref.indexOf('#')) : '';
      query.set('lang', language);
      link.setAttribute('href', path + '?' + query.toString() + hash);
    });
    try { localStorage.setItem('hccepose-language', language); } catch (error) {}
    if (updateUrl) {
      var current = new URL(location.href);
      current.searchParams.set('lang', language);
      history.replaceState(null, '', current.pathname + current.search + current.hash);
    }
  }
  languageButtons.forEach(function (button) {
    button.addEventListener('click', function () { applyLanguage(button.dataset.language, true); });
  });
  applyLanguage(root.dataset.lang, false);

  themeButton.setAttribute('aria-pressed', String(root.dataset.theme === 'dark'));
  themeButton.addEventListener('click', function () {
    var theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = theme;
    themeButton.setAttribute('aria-pressed', String(theme === 'dark'));
    try { localStorage.setItem('hccepose-theme', theme); } catch (error) {}
  });

  // Mobile disclosure leaves normal keyboard navigation intact; Escape returns focus.
  // 移动端导航保留正常 Tab 顺序，Escape 关闭后将焦点还给展开按钮。
  function closeMenu(returnFocus) {
    navigation.classList.remove('is-open');
    menuButton.setAttribute('aria-expanded', 'false');
    updateMenuLabel();
    if (returnFocus) menuButton.focus();
  }
  menuButton.addEventListener('click', function () {
    var open = menuButton.getAttribute('aria-expanded') !== 'true';
    navigation.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    updateMenuLabel();
    if (open) navigation.querySelector('a').focus();
  });
  navigation.querySelectorAll('a').forEach(function (link) {
    link.addEventListener('click', function () { closeMenu(false); });
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
      event.preventDefault();
      closeMenu(true);
    }
  });
  document.addEventListener('click', function (event) {
    if (!event.target.closest('.lp-header') && menuButton.getAttribute('aria-expanded') === 'true') closeMenu(false);
  });
  document.addEventListener('focusin', function (event) {
    if (!event.target.closest('.lp-header') && menuButton.getAttribute('aria-expanded') === 'true') closeMenu(false);
  });
  window.matchMedia('(min-width: 761px)').addEventListener('change', function (event) {
    if (event.matches) closeMenu(false);
  });

  // The selected video example opens automatically; native controls remain available.
  // 选中视频案例时自动展开播放，仍可用原生控件暂停、拖动进度或全屏。
  function setVideoPlaying(playing) {
    videoPlaying = playing;
    if (playing) {
      // The shared player shows loading, buffering, and retry feedback inside the video.
      // 共用播放器在画面内显示加载、缓冲和重试状态，避免首页出现第二份小字提示。
      videoElement.play().catch(function () {});
    } else {
      videoElement.pause();
    }
  }
  // Visibility pauses playback in video.js while retaining the open example.
  // 切后台时由 video.js 暂停并保留展开状态，返回后可继续自动播放。

  // Roving focus follows the WAI-ARIA tab convention: arrows, Home, and End.
  // 案例标签采用 WAI-ARIA 焦点约定：方向键切换，Home/End 跳至首尾。
  function selectTab(selectedTab, moveFocus) {
    caseTabs.style.setProperty('--case-index', String(tabs.indexOf(selectedTab)));
    tabs.forEach(function (tab) {
      var active = tab === selectedTab;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      var panel = document.getElementById(tab.getAttribute('aria-controls'));
      panel.hidden = !active;
      // Hidden panels reserve layout tracks but cannot receive focus or start video playback.
      // 隐藏案例保留布局占位，但退出焦点和辅助技术，并阻止后台视频播放。
      panel.inert = !active;
      panel.setAttribute('aria-hidden', String(!active));
    });
    if (selectedTab.id === 'tab-video' && !videoPlaying) setVideoPlaying(true);
    else if (selectedTab.id !== 'tab-video' && videoPlaying) setVideoPlaying(false);
    if (moveFocus) selectedTab.focus();
  }
  tabs.forEach(function (tab, index) {
    tab.addEventListener('click', function () { selectTab(tab, false); });
    tab.addEventListener('keydown', function (event) {
      var nextIndex = index;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') nextIndex = (index + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') nextIndex = 0;
      else if (event.key === 'End') nextIndex = tabs.length - 1;
      else return;
      event.preventDefault();
      selectTab(tabs[nextIndex], true);
    });
  });
}());
