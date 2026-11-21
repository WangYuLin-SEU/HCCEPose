/* Native players show each complete HD input or inference video.
   使用原生播放器展示完整高清输入或推理视频。 */
(function () {
  'use strict';

  var root = document.documentElement;
  var inputs = Array.from(document.querySelectorAll('video'));
  if (!inputs.length) return;
  var pageActive = true;
  var inputStates = [];

  function activeLanguage(state) {
    return !state.section || state.section.classList.contains(root.dataset.lang || 'zh');
  }

  function inputVisible(state) {
    if (!pageActive || document.hidden || !activeLanguage(state)) return false;
    if (state.video.closest('[hidden], [aria-hidden="true"]')) return false;
    // Some browsers retain layout rectangles inside closed details. Only the first
    // summary remains visible there, so check this native disclosure explicitly.
    // 部分浏览器在 details 收起后仍保留内部布局矩形；此时仅首个 summary 可见，
    // 因而显式检查原生折叠状态，避免加载收起的播放器。
    var ancestor = state.video.parentElement;
    while (ancestor) {
      if (ancestor.tagName === 'DETAILS' && !ancestor.open) {
        var summary = Array.from(ancestor.children).find(function (child) { return child.tagName === 'SUMMARY'; });
        if (!summary || !summary.contains(state.video)) return false;
      }
      ancestor = ancestor.parentElement;
    }
    // Load and play videos on screen; offscreen demos must not compete for bandwidth.
    // 只加载并播放视口中的视频，避免页面下方的演示同时争抢带宽。
    if (!state.video.getClientRects().length) return false;
    var visibility = getComputedStyle(state.video).visibility;
    if (visibility === 'hidden' || visibility === 'collapse') return false;
    var bounds = state.video.getBoundingClientRect();
    return bounds.width > 0 && bounds.height > 0 && bounds.bottom > 0 && bounds.right > 0 &&
      bounds.top < window.innerHeight && bounds.left < window.innerWidth;
  }

  function syncInputVisibility() {
    inputStates.forEach(function (state) {
      state.syncVisibility();
    });
  }

  function bilingualLabel(element) {
    // Both translations reserve their space; the shared language styles reveal one.
    // 两种译文共同占位，由共享语言样式显示当前语言，切换时不改变卡片几何。
    var existingPair = element.querySelector(':scope > .language-pair');
    if (existingPair) return {zh: existingPair.querySelector('.zh'), en: existingPair.querySelector('.en')};
    var pair = document.createElement('span');
    pair.className = 'language-pair';
    var zh = document.createElement('span');
    zh.className = 'zh';
    zh.lang = 'zh-CN';
    var en = document.createElement('span');
    en.className = 'en';
    en.lang = 'en';
    pair.append(zh, en);
    element.appendChild(pair);
    return {zh: zh, en: en};
  }

  // Every rendered video starts muted and loops. JavaScript gates playback so native
  // autoplay cannot start hidden language copies before their source has been prepared.
  // 所有已渲染的视频统一静音循环，由脚本检查显隐后起播，避免原生 autoplay
  // 在资源准备前启动隐藏语言副本。
  inputs.forEach(function (video) {
    video.muted = true;
    video.defaultMuted = true;
    video.loop = true;
    video.playsInline = true;
    var source = video.querySelector('source');
    var sourceUrl = video.getAttribute('src') || (source && source.getAttribute('src'));
    if (!sourceUrl) return;
    var nativePlay = video.play.bind(video);
    var nativePause = video.pause.bind(video);
    // Reuse the initial HTML loading card so it is visible before this script arrives.
    // Bare videos keep the same fallback wrapper without cropping or redrawing the video.
    // 复用初始 HTML 中的加载卡片，使脚本到达前已有提示；裸 video 仍按相同结构
    // 添加包装层，不裁切或重绘视频。
    var wrapper = video.parentElement;
    if (!wrapper.classList.contains('video-player')) {
      wrapper = document.createElement('div');
      wrapper.className = 'video-player';
      video.parentElement.insertBefore(wrapper, video);
      wrapper.appendChild(video);
    }
    wrapper.hidden = video.hidden;
    var overlay = wrapper.querySelector(':scope > .video-loading');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.className = 'video-loading';
      var card = document.createElement('div');
      card.className = 'video-loading-card';
      var newSpinner = document.createElement('span');
      newSpinner.className = 'video-loading-spinner';
      var copy = document.createElement('div');
      copy.className = 'video-loading-copy';
      var newTitle = document.createElement('strong');
      newTitle.className = 'video-loading-title';
      var newDetail = document.createElement('span');
      newDetail.className = 'video-loading-detail';
      var newRetry = document.createElement('button');
      newRetry.className = 'video-loading-retry';
      copy.append(newTitle, newDetail, newRetry);
      card.append(newSpinner, copy);
      overlay.appendChild(card);
      wrapper.appendChild(overlay);
    }
    overlay.setAttribute('role', 'status');
    overlay.setAttribute('aria-live', 'polite');
    overlay.setAttribute('aria-atomic', 'true');
    var spinner = overlay.querySelector('.video-loading-spinner');
    spinner.setAttribute('aria-hidden', 'true');
    var title = overlay.querySelector('.video-loading-title');
    var detail = overlay.querySelector('.video-loading-detail');
    var retry = overlay.querySelector('.video-loading-retry');
    retry.type = 'button';
    var titleLabel = bilingualLabel(title);
    var detailLabel = bilingualLabel(detail);
    var retryLabel = bilingualLabel(retry);
    var state = {
      video: video, wrapper: wrapper, section: video.closest('.doc-language'), sourceUrl: new URL(sourceUrl, document.baseURI).href,
      sourceReady: false, playPromise: null,
      wantsPlay: true, hasPlayed: false, notice: 'loading', failure: null,
      ignoredPauses: 0, syncVisibility: syncVisibility
    };
    inputStates.push(state);

    function syncWrapper() {
      // Copy hidden only when it changed; our own observer must not create a mutation loop.
      // 仅在值变化时同步 hidden，避免自身观察器产生循环；首页关闭后不留空位。
      if (wrapper.hidden !== video.hidden) wrapper.hidden = video.hidden;
    }

    function renderOverlay() {
      var notice = state.notice;
      var actionable = notice === 'error' || notice === 'blocked';
      var visible = Boolean(notice) && inputVisible(state) && (state.wantsPlay || actionable);
      if (overlay.hidden === visible) overlay.hidden = !visible;
      if (notice) overlay.dataset.state = notice;
      else delete overlay.dataset.state;
      spinner.hidden = actionable;
      retry.hidden = !actionable;
      var titleZh = '视频加载中';
      var titleEn = 'Loading video';
      var detailZh = '即将自动播放';
      var detailEn = 'Starting shortly';
      if (notice === 'buffering') {
        titleZh = '正在缓冲';
        titleEn = 'Buffering video';
        detailZh = '正在等待视频数据';
        detailEn = 'Waiting for video data';
      } else if (notice === 'error') {
        titleZh = '视频暂时无法加载';
        titleEn = 'Video unavailable';
        detailZh = '请检查网络后重试';
        detailEn = 'Check your connection and try again';
      } else if (notice === 'blocked') {
        titleZh = '自动播放已暂停';
        titleEn = 'Autoplay paused';
        detailZh = '点击“播放”开始观看';
        detailEn = 'Select Play to start the video';
      }
      var retryZh = notice === 'blocked' ? '播放' : '重试';
      var retryEn = notice === 'blocked' ? 'Play' : 'Retry';
      // Update both labels only when the state changes, without repeated live announcements.
      // 状态变化时同时更新双语文字；文字相同则不重写，避免读屏重复播报。
      if (titleLabel.zh.textContent !== titleZh) titleLabel.zh.textContent = titleZh;
      if (titleLabel.en.textContent !== titleEn) titleLabel.en.textContent = titleEn;
      if (detailLabel.zh.textContent !== detailZh) detailLabel.zh.textContent = detailZh;
      if (detailLabel.en.textContent !== detailEn) detailLabel.en.textContent = detailEn;
      if (retryLabel.zh.textContent !== retryZh) retryLabel.zh.textContent = retryZh;
      if (retryLabel.en.textContent !== retryEn) retryLabel.en.textContent = retryEn;
      if (visible && !actionable) video.setAttribute('aria-busy', 'true');
      else video.removeAttribute('aria-busy');
      if (!visible && document.activeElement === retry && inputVisible(state)) video.focus({preventScroll: true});
    }

    function showNotice(notice) {
      state.notice = notice;
      renderOverlay();
    }

    function failPlayback(kind) {
      state.failure = kind;
      state.wantsPlay = false;
      showNotice(kind);
      nativePause();
    }

    function resetFailure() {
      if (state.failure === 'error') {
        // Reassign the native source only after an explicit retry of a failed player.
        // 仅在明确重试失败播放器时重新设置原生源，正常暂停/恢复不重载视频。
        state.sourceReady = false;
        state.hasPlayed = false;
      }
      state.failure = null;
    }

    function syncVisibility() {
      syncWrapper();
      renderOverlay();
      if (!inputVisible(state)) {
        // Hiding preserves automatic intent; an explicit pause clears it below.
        // 隐藏暂停保留自动播放意图；下方显式 pause 才取消意图，避免恢复后顶掉手动暂停。
        if (!video.paused) {
          state.ignoredPauses += 1;
          nativePause();
        }
        return;
      }
      if (state.wantsPlay && !state.failure && video.paused) requestPlayback().catch(function () {});
    }

    function prepareInput() {
      syncWrapper();
      // Focusing a failed player is not a retry. Keep the error and its action visible
      // until an explicit play resets it; otherwise focus can hide the card without intent.
      // 失败后聚焦播放器不等于重试；只有明确播放才清除失败，避免焦点预加载
      // 将错误卡片隐藏，却没有播放意图，最终停在已下载但未播放的状态。
      if (state.failure) return Promise.resolve();
      if (state.sourceReady || !inputVisible(state)) return Promise.resolve();
      showNotice('loading');
      // Faststart MP4s can decode while bytes arrive, including HTTP 200 responses.
      // Leave streaming, buffering and seeking to the native player instead of waiting
      // for a complete fetch/Blob; keep the card until the first playing event.
      // faststart MP4 可边接收边解码，包括 HTTP 200；交由原生播放器处理流式下载、
      // 缓冲与拖动，不再等待 fetch/Blob 整段完成，提示保留到首次 playing 事件。
      state.sourceReady = true;
      video.src = state.sourceUrl;
      return Promise.resolve();
    }

    function playPreparedInput() {
      return prepareInput().then(function () {
        if (!state.sourceReady || !state.wantsPlay || !inputVisible(state)) {
          throw new DOMException('Video playback was cancelled.', 'AbortError');
        }
        return nativePlay();
      });
    }

    function requestPlayback() {
      // The homepage can explicitly play while its visibility observer is also starting.
      // 首页显式 play 与可见性观察可能同时启动，共用一次准备和播放 Promise。
      if (state.playPromise) return state.playPromise;
      if (!state.hasPlayed) showNotice('loading');
      state.playPromise = playPreparedInput();
      state.playPromise.then(function () {
        state.playPromise = null;
        syncVisibility();
      }, function (error) {
        state.playPromise = null;
        if (error.name !== 'AbortError') failPlayback(error.name === 'NotAllowedError' ? 'blocked' : 'error');
        syncVisibility();
      });
      return state.playPromise;
    }

    // A page may close its player by calling pause() while preparation has already paused
    // the native element. Remember that explicit call even when no pause event is emitted.
    // 页面关闭播放器时可能再次调用已暂停元素的 pause()；此时浏览器不发 pause 事件，
    // 因而只在本实例上记录 play/pause 意图，避免关闭后异步加载完成又开始播放。
    video.play = function () { resetFailure(); state.wantsPlay = true; syncWrapper(); return requestPlayback(); };
    video.pause = function () { state.wantsPlay = false; syncWrapper(); showNotice(null); nativePause(); };
    retry.addEventListener('click', function () { video.play().catch(function () {}); });
    video.addEventListener('pointerdown', function () { prepareInput().catch(function () {}); });
    video.addEventListener('focusin', function () { prepareInput().catch(function () {}); });
    video.addEventListener('play', function () {
      // Native controls bypass the instance's play() wrapper. A live play event is also
      // an explicit retry; a queued event after pause must not revive cancelled intent.
      // 原生控件不经过实例 play() 包装；有效的 play 事件同样明确重试，
      // 但已暂停后才送达的旧事件不能重新激活已取消的意图。
      if (video.paused) return;
      resetFailure();
      if (state.sourceReady) {
        state.wantsPlay = true;
        syncVisibility();
        return;
      }
      // Native Play confirms intent before this instance has assigned its source.
      // 本实例设置原生源之前，原生播放键也明确恢复播放意图。
      state.wantsPlay = true;
      if (!video.paused) {
        state.ignoredPauses += 1;
        nativePause();
      }
      if (state.wantsPlay) requestPlayback().catch(function () {});
    });
    video.addEventListener('pause', function () {
      if (state.ignoredPauses > 0) { state.ignoredPauses -= 1; renderOverlay(); return; }
      state.wantsPlay = false;
      if (!state.failure) showNotice(null);
      else renderOverlay();
    });
    video.addEventListener('playing', function () {
      if (video.paused) return;
      state.hasPlayed = true;
      showNotice(null);
    });
    ['waiting', 'stalled', 'seeking'].forEach(function (name) {
      video.addEventListener(name, function () {
        // Network delivery may stall while buffered frames still play normally.
        // 网络暂时停传时，已有缓冲帧可能仍正常播放，不在这种情况下遮住画面。
        if (name === 'stalled' && video.readyState >= 3 && !video.paused && !video.seeking) return;
        if (state.wantsPlay && !state.failure) showNotice(state.hasPlayed ? 'buffering' : 'loading');
      });
    });
    ['seeked', 'canplay'].forEach(function (name) {
      video.addEventListener(name, function () {
        if (state.failure) return;
        if (!state.wantsPlay || (state.hasPlayed && !video.paused && !video.seeking && video.readyState >= 3)) showNotice(null);
      });
    });
    video.addEventListener('error', function () { failPlayback('error'); });
    video.querySelectorAll('source').forEach(function (item) { item.addEventListener('error', function () { failPlayback('error'); }); });
    video.addEventListener('ended', function () { state.wantsPlay = false; showNotice(null); });
  });

  // Only a player or its ancestors can change its rendered visibility. Ignore unrelated
  // hero animation styles so they do not recalculate every player's layout on each frame.
  // 仅播放器及其祖先的变化影响显隐；忽略无关横幅动画样式，避免每帧重算所有视频布局。
  new MutationObserver(function (records) {
    var relevant = records.some(function (record) {
      return inputStates.some(function (state) {
        if (record.target === state.wrapper && record.attributeName === 'hidden' && state.wrapper.hidden === state.video.hidden) return false;
        return record.target.contains(state.video);
      });
    });
    if (relevant) syncInputVisibility();
  }).observe(root, {attributes: true, subtree: true, attributeFilter: ['data-lang', 'data-banner-restore', 'hidden', 'aria-hidden', 'class', 'style', 'open']});
  // Scrolling preserves playback intent: returning resumes automatic videos, while a
  // manual pause remains paused. Hidden-language players still fail the checks above.
  // 滚动离开保留播放意图：返回后自动续播，手动暂停则保持；隐藏语言仍由上方检查拦截。
  if ('IntersectionObserver' in window) {
    var viewportObserver = new IntersectionObserver(syncInputVisibility);
    inputStates.forEach(function (state) { viewportObserver.observe(state.wrapper); });
  } else {
    var visibilityFrame = 0;
    window.addEventListener('scroll', function () {
      if (visibilityFrame) return;
      visibilityFrame = requestAnimationFrame(function () {
        visibilityFrame = 0;
        syncInputVisibility();
      });
    }, {passive: true});
  }
  window.addEventListener('resize', syncInputVisibility);
  document.addEventListener('visibilitychange', syncInputVisibility);
  window.addEventListener('pagehide', function () { pageActive = false; syncInputVisibility(); });
  window.addEventListener('pageshow', function () { pageActive = true; syncInputVisibility(); });
  syncInputVisibility();
}());
