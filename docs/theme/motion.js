// Continuous silk-like surfaces for the homepage and documentation banners.
// 首页与文档横幅共用的连续柔性曲面；保留原图作为静态与失败回退。
(function () {
  'use strict';

  var containers = Array.from(document.querySelectorAll('.pose-motion'));
  if (!containers.length) return;

  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var forcedColors = window.matchMedia('(forced-colors: active)');
  var finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  // Two quarter-shifted coordinate phases keep flowing when either axis turns around.
  // 两个相差四分之一周期的坐标相位，让单轴转向时另一轴仍连续流动。
  var phasePropertiesReady = false;
  if (window.CSS && typeof window.CSS.registerProperty === 'function') {
    try {
      window.CSS.registerProperty({name: '--code-flow-x', syntax: '<number>', inherits: true, initialValue: '1'});
      window.CSS.registerProperty({name: '--code-flow-y', syntax: '<number>', inherits: true, initialValue: '0'});
      phasePropertiesReady = true;
    } catch (error) {
      // Older engines retain the ordinary gradient keyframes and SVG flow.
      // 旧引擎保留普通渐变关键帧与 SVG 流动作为回退。
    }
  }

  function prepareTitleTails(index) {
    // All eight glyphs emit broad ribbons and fine branching filaments. Random targets are
    // chosen per segment, then eased; no frame-to-frame random jitter is introduced.
    // 八个字母各延伸宽彩带与细分叉；每段只采样一次随机目标，再平滑过渡，避免逐帧随机抖动。
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('pose-title-tails');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    var prefix = 'pose-title-tail-' + index;
    var defs = '<defs><filter id="' + prefix + '-glow" x="-12%" y="-65%" width="124%" height="230%"><feGaussianBlur stdDeviation="3.5"/></filter>';
    var palettes = [
      ['#50ffd8', '#778dff', '#ff74dc'],
      ['#63b9ff', '#b67aff', '#53ffe3'],
      ['#ff7bd5', '#61ddff', '#b49aff'],
      ['#ffe29a', '#ff857b', '#f57bff'],
      ['#b7ffb0', '#4eeaff', '#9287ff'],
      ['#c193ff', '#ff83d1', '#7af7ff']
    ];
    palettes.forEach(function (palette, colorIndex) {
      defs += '<linearGradient id="' + prefix + '-color-' + colorIndex + '" x1="0" y1="0" x2="1" y2="0">';
      [0, 0.1, 0.3, 0.56, 0.82, 1].forEach(function (offset, stopIndex) {
        var colors = palette.slice(stopIndex % 3).concat(palette.slice(0, stopIndex % 3));
        colors.push(colors[0]);
        defs += '<stop offset="' + offset + '" stop-color="' + colors[0] + '" stop-opacity="' + [0, 0.28, 0.76, 0.6, 0.23, 0][stopIndex] + '">' +
          '<animate attributeName="stop-color" values="' + colors.join(';') + '" dur="' + (8.4 + colorIndex * 0.8) + 's" repeatCount="indefinite" calcMode="spline" keyTimes="0;0.333333;0.666667;1" keySplines=".42 0 .58 1;.42 0 .58 1;.42 0 .58 1"/></stop>';
      });
      defs += '</linearGradient>';
    });
    // A broad, low-density veil extends beneath the whole word. Opaque glyph fills
    // cover the rear light naturally; no shared vertical gate starts at the final letter.
    // 宽缓的低密度渐隐区贯穿整词后方，由不透明字形自然遮住底层光，不再从末字统一开闸。
    defs += '<linearGradient id="' + prefix + '-rear-veil" gradientUnits="userSpaceOnUse" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="white" stop-opacity="0.12"/><stop offset="0.35" stop-color="white" stop-opacity="0.18"/><stop offset="0.7" stop-color="white" stop-opacity="0.42"/><stop offset="1" stop-color="white" stop-opacity="1"/></linearGradient>' +
      '<mask id="' + prefix + '-rear-density" maskUnits="userSpaceOnUse" style="mask-type:alpha"><rect class="pose-tail-veil-rect" fill="url(#' + prefix + '-rear-veil)"/></mask>';
    var mask = ' mask="url(#' + prefix + '-rear-density)"';
    svg.innerHTML = defs + '</defs><g class="pose-tail-glow"' + mask + ' filter="url(#' + prefix + '-glow)"></g><g class="pose-tail-core"' + mask + '></g><g class="pose-tail-filaments"' + mask + '></g><g class="pose-glyph-orbits"></g>';
    var glow = svg.querySelector('.pose-tail-glow');
    var core = svg.querySelector('.pose-tail-core');
    var filaments = svg.querySelector('.pose-tail-filaments');
    var ribbons = [];
    for (var glyph = 0; glyph < 8; glyph += 1) {
      for (var strand = 0; strand < 3; strand += 1) {
        var id = prefix + '-ribbon-' + glyph + '-' + strand;
        var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.id = id;
        path.setAttribute('fill', 'url(#' + prefix + '-color-' + (glyph + strand) % palettes.length + ')');
        path.setAttribute('opacity', ['0.68', '0.58', '0.82'][strand]);
        (strand < 2 ? core : filaments).appendChild(path);
        // Broad sheets provide the bloom; duplicating every fine branch into the blur
        // adds paint cost without making its already luminous outline more readable.
        // 宽彩带负责柔光；不将每条细分叉重复送入模糊层，保留清晰层次并减少绘制开销。
        if (strand === 0) {
          var halo = document.createElementNS('http://www.w3.org/2000/svg', 'use');
          halo.setAttribute('href', '#' + id);
          glow.appendChild(halo);
        }
        ribbons.push({
          path: path, glyph: glyph, strand: strand, phase: Math.random() * Math.PI * 2, speed: 0.5 + Math.random() * 0.65,
          from: [Math.random(), Math.random(), Math.random(), Math.random(), Math.random()],
          to: [Math.random(), Math.random(), Math.random(), Math.random(), Math.random()],
          startedAt: 0, duration: 4 + Math.random() * 3
        });
      }
    }
    return {svg: svg, palettes: palettes, ribbons: ribbons,
      veil: svg.querySelector('#' + prefix + '-rear-veil'), veilRect: svg.querySelector('.pose-tail-veil-rect'),
      frame: null, previousTimestamp: null, elapsed: 0, geometry: null};
  }

  function prepareGlyphWeave(tails) {
    var namespace = 'http://www.w3.org/2000/svg';
    var front = document.createElementNS(namespace, 'svg');
    front.classList.add('pose-title-front-orbits');
    front.setAttribute('aria-hidden', 'true');
    front.setAttribute('focusable', 'false');
    var orbits = tails.svg.querySelector('.pose-glyph-orbits');
    var events = [];
    for (var glyph = 0; glyph < 8; glyph += 1) {
      var paths = [];
      // Keep the two rear turns, cast-off thread and faint front glint local to each
      // letter. Their independent timing never deforms or recolors the background.
      // 每字保留两段后绕弧、甩出细丝与极淡前掠光；独立节奏不再扭曲或改染背景。
      [orbits, orbits, orbits, front].forEach(function (group, pathIndex) {
        var path = document.createElementNS(namespace, 'path');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', tails.palettes[(glyph + pathIndex) % tails.palettes.length][pathIndex % 3]);
        path.setAttribute('stroke-width', '0.8');
        path.setAttribute('opacity', '0');
        path.dataset.glyph = String(glyph);
        group.appendChild(path);
        paths.push(path);
      });
      var duration = 7 + Math.random() * 4;
      events.push({glyph: glyph, paths: paths, phase: Math.random() * Math.PI * 2,
        startedAt: -duration * (glyph / 8), duration: duration,
        side: glyph % 2 === 0 ? -1 : 1, energy: 0});
    }
    // A single faint front pass has its own random quiet interval and smooth envelope.
    // 一次只保留一道极淡前掠光，独立随机安排静默间隔，并用平滑包络进出。
    return {front: front, events: events, frontPass: {glyph: Math.floor(Math.random() * 8),
      startedAt: 0.8 + Math.random() * 2, duration: 2.6 + Math.random() * 2, phase: Math.random() * Math.PI * 2}};
  }

  function easedUnit(value) {
    var t = Math.max(0, Math.min(1, value));
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  // Elliptic turns use exact cubic tangents; partial turns avoid eight rigid rings.
  // 椭圆转折按精确切向量拼接三次曲线，只画局部绕弧，避免变成八个呆板圆环。
  function orbitPath(cx, cy, rx, ry, tilt, start, sweep) {
    var segments = 5;
    var step = sweep / segments;
    var tangentScale = 4 / 3 * Math.tan(step / 4);
    function sample(angle) {
      var x = rx * Math.cos(angle);
      var y = ry * Math.sin(angle);
      return [cx + x, cy + y + x * tilt, -rx * Math.sin(angle), ry * Math.cos(angle) - rx * Math.sin(angle) * tilt];
    }
    var point = sample(start);
    var path = 'M' + point[0].toFixed(2) + ' ' + point[1].toFixed(2);
    for (var segment = 0; segment < segments; segment += 1) {
      var next = sample(start + step * (segment + 1));
      path += ' C' + [point[0] + point[2] * tangentScale, point[1] + point[3] * tangentScale,
        next[0] - next[2] * tangentScale, next[1] - next[3] * tangentScale, next[0], next[1]].map(function (value) { return value.toFixed(2); }).join(' ');
      point = next;
    }
    return path;
  }

  function renderGlyphWeave(title, geometry, time) {
    var frontPass = title.weave.frontPass;
    if (time >= frontPass.startedAt + frontPass.duration) {
      frontPass.startedAt = time + 2 + Math.random() * 4;
      frontPass.duration = 2.6 + Math.random() * 2;
      frontPass.glyph = Math.floor(Math.random() * 8);
      frontPass.phase = Math.random() * Math.PI * 2;
    }
    var frontProgress = Math.max(0, Math.min(1, (time - frontPass.startedAt) / frontPass.duration));
    var frontEnvelope = Math.pow(Math.sin(Math.PI * frontProgress), 4);
    title.weave.events.forEach(function (event) {
      if (time >= event.startedAt + event.duration) {
        event.startedAt += event.duration;
        event.duration = 7 + Math.random() * 4;
        event.side = Math.random() > 0.5 ? -1 : 1;
      }
      var cycle = (time - event.startedAt) / event.duration;
      var outward = easedUnit(cycle / 0.55);
      var outboundEnergy = cycle < 0.55 ? Math.pow(Math.sin(Math.PI * Math.max(0, cycle / 0.55)), 2) : 0;
      var returnEnergy = cycle > 0.52 && cycle < 0.95 ? Math.pow(Math.sin(Math.PI * (cycle - 0.52) / 0.43), 2) : 0;
      event.energy = outboundEnergy;
      var anchor = geometry.anchors[event.glyph];
      var font = geometry.fontSize;
      var orbitTime = time * (0.48 + event.glyph * 0.017) + event.phase;
      var cx = anchor.center;
      var cy = geometry.paddingY + geometry.height * 0.52;
      var rx = anchor.width * (0.6 + 0.08 * Math.sin(orbitTime * 0.63)) + font * 0.06;
      var ry = font * (0.33 + 0.055 * Math.sin(orbitTime * 0.71));
      var tilt = -0.4 + 0.27 * Math.sin(orbitTime * 0.37);
      event.paths[0].setAttribute('d', orbitPath(cx, cy, rx, ry, tilt, orbitTime, Math.PI * 1.16));
      event.paths[1].setAttribute('d', orbitPath(cx, cy, rx * 1.1, ry * 0.8, -tilt, -orbitTime * 0.81 + 1.8, Math.PI * 0.78));
      event.paths[0].setAttribute('opacity', (0.24 + outboundEnergy * 0.32).toFixed(3));
      event.paths[1].setAttribute('opacity', (0.12 + returnEnergy * 0.25).toFixed(3));
      // Only a thin thread may curl over the front; broad color always stays behind.
      // 仅细丝偶尔绕到字前，最高 4.5%；宽彩带始终留在字后，维持字形可读。
      var frontAngle = frontPass.phase + frontProgress * Math.PI * 1.3;
      event.paths[3].setAttribute('d', orbitPath(cx, cy, rx, ry, tilt, frontAngle, Math.PI * 0.45));
      event.paths[3].setAttribute('opacity', (event.glyph === frontPass.glyph ? 0.045 * frontEnvelope : 0).toFixed(4));
      var releaseY = cy + event.side * font * 0.43;
      var castLength = (geometry.width + geometry.extraWidth * (0.4 + outward * 0.6) - cx);
      var castEnd = cx + castLength;
      var castY = cy + event.side * font * 0.5 + Math.sin(orbitTime * 0.7) * font * 0.38;
      event.paths[2].setAttribute('d', 'M' + cx.toFixed(2) + ' ' + releaseY.toFixed(2) + ' C' +
        [cx + castLength * 0.22, releaseY + event.side * font * 0.44, cx + castLength * 0.64, castY, castEnd, castY].map(function (value) { return value.toFixed(2); }).join(' '));
      event.paths[2].setAttribute('opacity', (outboundEnergy * 0.36).toFixed(3));
    });
  }

  function renderTitleTails(title, timestamp) {
    var tails = title.tails;
    tails.frame = null;
    if (title.element.closest('.pose-motion').dataset.titleMotion !== 'running') return;
    if (tails.previousTimestamp !== null) tails.elapsed += (timestamp - tails.previousTimestamp) / 1000;
    tails.previousTimestamp = timestamp;
    // One active-time clock advects every text layer and the ribbons continuously.
    // 单一可见时间同时推动文字各层与尾焰；正弦/余弦连续可导，不在关键帧节点突然换向。
    var time = tails.elapsed;
    var geometry = tails.geometry;
    if (geometry) renderGlyphWeave(title, geometry, time);
    title.element.style.setProperty('--code-flow-x', Math.cos(time * 0.44).toFixed(5));
    title.element.style.setProperty('--code-flow-y', Math.sin(time * 0.44).toFixed(5));
    title.flowOffset.setAttribute('dx', (12 * Math.sin(time * 0.27)).toFixed(3));
    title.flowOffset.setAttribute('dy', (10 * Math.cos(time * 0.31)).toFixed(3));
    if (geometry) {
      tails.ribbons.forEach(function (ribbon) {
        if (tails.elapsed >= ribbon.startedAt + ribbon.duration) {
          ribbon.from = ribbon.to;
          ribbon.to = [Math.random(), Math.random(), Math.random(), Math.random(), Math.random()];
          ribbon.startedAt = tails.elapsed;
          ribbon.duration = 4 + Math.random() * 3;
        }
        var progress = Math.max(0, Math.min(1, (tails.elapsed - ribbon.startedAt) / ribbon.duration));
        var eased = progress * progress * progress * (progress * (progress * 6 - 15) + 10);
        var lengthChoice = ribbon.from[0] + (ribbon.to[0] - ribbon.from[0]) * eased;
        var curlChoice = ribbon.from[1] + (ribbon.to[1] - ribbon.from[1]) * eased;
        var widthChoice = ribbon.from[2] + (ribbon.to[2] - ribbon.from[2]) * eased;
        var rootChoice = ribbon.from[3] + (ribbon.to[3] - ribbon.from[3]) * eased;
        var rootHeightChoice = ribbon.from[4] + (ribbon.to[4] - ribbon.from[4]) * eased;
        var anchor = geometry.anchors[ribbon.glyph];
        var castEnergy = title.weave.events[ribbon.glyph].energy;
        // Independent roots drift left through the word's rear plane. Both ends taper
        // to transparency, so a ribbon feels like a passing current rather than a nozzle.
        // 各条根部在整词后方独立向左游移，两端收细并透明，呈现穿行气流而非集中喷口。
        var x = anchor.x - (anchor.x + geometry.paddingLeft) * (0.18 + rootChoice * 0.68);
        var y = geometry.paddingY + geometry.height * 0.52 + (rootHeightChoice - 0.5) * geometry.fontSize * 0.4;
        y += Math.sin(time * 0.19 + ribbon.phase) * geometry.fontSize * 0.06;
        var availableLength = geometry.width + geometry.extraWidth - x;
        var length = geometry.narrow ? availableLength * (0.52 + lengthChoice * 0.48) :
          Math.min(availableLength, geometry.fontSize * (6.8 + ribbon.glyph * 0.25 + lengthChoice * 2.5)) * (0.85 + lengthChoice * 0.15);
        length *= 0.76 + castEnergy * 0.24;
        var curl = (curlChoice - 0.5) * geometry.fontSize * 1.05;
        var phase = ribbon.phase;
        var breadth = geometry.fontSize * (ribbon.strand < 2 ? 0.21 + ribbon.strand * 0.025 : 0.04) * (0.8 + widthChoice * 0.65) * (geometry.narrow ? 0.72 : 1);
        ribbon.path.setAttribute('opacity', ((ribbon.strand < 2 ? 0.54 : 0.74) * (0.35 + castEnergy * 0.65)).toFixed(3));
        var travel = time * ribbon.speed;
        var y1 = y - curl * 0.4 + Math.sin(travel + phase) * geometry.fontSize * 0.25;
        var middleY = y + curl * 0.35 + Math.sin(travel * 0.92 + phase * 1.3) * geometry.fontSize * 0.3;
        var y2 = y + curl + Math.sin(travel * 1.14 + phase * 1.7) * geometry.fontSize * 0.44;
        var fan = (ribbon.strand - 1) * geometry.fontSize * 0.24;
        var tipY = y + fan + curl * 0.45 + Math.sin(travel * 0.84 + phase * 0.7) * geometry.fontSize * 0.3;
        // Two joined cubic sections give each edge a flowing shoulder and a second curl.
        // Broad colored sheets and thin branches fan out, sharing a softly disappearing tip.
        // 每条边用两段相接三次曲线形成柔性肩部与二次卷曲；宽彩带与细分叉展开，尾端柔和消散。
        var points = [x, y,
          x + length * 0.16, y1 - breadth * 0.55, x + length * 0.32, middleY - breadth * 0.5, x + length * 0.5, middleY - breadth * 0.32,
          x + length * 0.68, middleY - breadth * 0.14, x + length * 0.86, y2, x + length, tipY,
          x + length * 0.86, y2, x + length * 0.68, middleY + breadth * 0.14, x + length * 0.5, middleY + breadth * 0.32,
          x + length * 0.32, middleY + breadth * 0.5, x + length * 0.16, y1 + breadth * 0.55, x, y
        ].map(function (value) { return value.toFixed(2); });
        ribbon.path.setAttribute('d', 'M' + points.slice(0, 2).join(' ') + ' C' + points.slice(2, 8).join(' ') + ' C' + points.slice(8, 14).join(' ') + ' C' + points.slice(14, 20).join(' ') + ' C' + points.slice(20).join(' ') + ' Z');
      });
    }
    tails.frame = window.requestAnimationFrame(function (next) { renderTitleTails(title, next); });
  }

  // H/C/C/E/P/o/s/e keep fixed HCCE levels 1–8. The first level is coordinates;
  // later levels fold at 1, 2, 4, 8, 16, 32 and 64 cycles over the coordinate domain.
  // 八个字母固定对应 HCCE 第 1–8 层：首层是坐标，其后在完整坐标域折返
  // 1、2、4、8、16、32、64 次。这里只表现编码结构，不作为预测结果或标签展示。
  function prepareTitle(container, index) {
    var title = container.querySelector('.pose-motion-title');
    if (!title || title.textContent.trim() !== 'HCCEPose') return null;
    var word = document.createElement('span');
    word.className = 'pose-title-word';
    word.textContent = title.textContent;
    var aura = document.createElement('span');
    aura.className = 'pose-title-aura';
    aura.textContent = word.textContent;
    aura.setAttribute('aria-hidden', 'true');
    var letters = document.createElement('span');
    letters.className = 'pose-code-letters';
    letters.setAttribute('aria-hidden', 'true');
    Array.from(word.textContent).forEach(function (character, index) {
      var letter = document.createElement('span');
      letter.className = 'pose-code-letter';
      letter.textContent = character;
      letter.dataset.character = character;
      letter.dataset.codeLevel = String(index + 1);
      letter.style.setProperty('--code-frequency', String(index === 0 ? 1 : Math.pow(2, index - 1)));
      letter.style.setProperty('--code-parent-frequency', String(index < 2 ? 1 : Math.pow(2, index - 2)));
      letters.appendChild(letter);
    });
    // Warp the colored fill inside its original alpha mask. The clear glyph edge and
    // intact native word remain stable while nested fields bend and flow inside them.
    // 在原字形透明度范围内弯曲彩色填充；编码场在字内柔性流动，清晰边缘及原词占位保持稳定。
    var filterId = 'pose-title-fluid-' + index;
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('pose-title-filter');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    svg.innerHTML = '<defs><filter id="' + filterId + '" x="-8%" y="-30%" width="116%" height="160%" color-interpolation-filters="sRGB">' +
      '<feTurbulence type="fractalNoise" baseFrequency="0.013 0.025" numOctaves="2" seed="11" result="field"/>' +
      '<feOffset in="field" dy="10" result="drifting-field"/>' +
      '<feDisplacementMap in="SourceGraphic" in2="drifting-field" scale="9" xChannelSelector="R" yChannelSelector="B" result="flowing-code"/>' +
      '<feComposite in="flowing-code" in2="SourceAlpha" operator="in" result="clipped-code"/>' +
      '<feBlend in="clipped-code" in2="SourceGraphic" mode="normal"/>' +
      '</filter></defs>';
    title.style.setProperty('--title-fluid-filter', 'url(#' + filterId + ')');
    title.dataset.phaseReady = String(phasePropertiesReady);
    var tails = prepareTitleTails(index);
    var weave = prepareGlyphWeave(tails);
    title.replaceChildren(word, tails.svg, aura, letters, weave.front, svg);
    return {element: title, word: word, letters: Array.from(letters.children), svg: svg,
      flowOffset: svg.querySelector('feOffset'), tails: tails, weave: weave};
  }

  function measureTitle(title) {
    if (!title) return;
    var bounds = title.word.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    var wordStyle = getComputedStyle(title.word);
    var layoutWidth = parseFloat(wordStyle.width);
    var layoutHeight = parseFloat(wordStyle.height);
    var localScaleX = layoutWidth / bounds.width;
    var localScaleY = layoutHeight / bounds.height;
    // Measure the intact word so kerning, hit areas and responsive width never change.
    // A common square coordinate domain extends beyond the glyphs; the finest level
    // stays resolved instead of collapsing to subpixel stripes on a small banner.
    // 测量完整原词，保留字距、交互占位和响应式宽度；共用的方形坐标域延展至字形外，
    // 让高层仍有可分辨的纹理，避免小横幅上的亚像素细线混叠。
    // Convert viewport measurements back to local CSS pixels, including browser zoom.
    // 将视口测量值还原为局部 CSS 像素，兼容缩放和响应式尺寸变化，保留文字本身的字号。
    title.element.style.setProperty('--code-field-size', (layoutWidth * 1.6).toFixed(3) + 'px');
    var range = document.createRange();
    var anchors = [];
    title.letters.forEach(function (letter, index) {
      range.setStart(title.word.firstChild, index);
      range.setEnd(title.word.firstChild, index + 1);
      var character = range.getBoundingClientRect();
      var left = (character.left - bounds.left) * localScaleX;
      letter.style.left = left.toFixed(3) + 'px';
      letter.style.width = (character.width * localScaleX).toFixed(3) + 'px';
      letter.style.height = (bounds.height * localScaleY).toFixed(3) + 'px';
      letter.style.setProperty('--code-origin-x', (-left).toFixed(3) + 'px');
      var width = character.width * localScaleX;
      anchors.push({x: left + width * 0.78, center: left + width * 0.5, width: width});
    });
    var fontSize = parseFloat(wordStyle.fontSize);
    var container = title.element.closest('.pose-motion');
    var containerBounds = container.getBoundingClientRect();
    var availableRight = (containerBounds.right - bounds.right - 16) * localScaleX;
    var paddingLeft = Math.max(0, Math.min(fontSize * 0.8, (bounds.left - containerBounds.left - 8) * localScaleX));
    var narrow = window.innerWidth <= 760;
    var extraWidth = Math.max(fontSize * 0.35, Math.min(fontSize * (narrow ? 3 : 8.5), availableRight));
    var paddingY = fontSize * 0.95;
    title.tails.geometry = {anchors: anchors, fontSize: fontSize, width: layoutWidth, height: layoutHeight,
      paddingY: paddingY, paddingLeft: paddingLeft, extraWidth: extraWidth, narrow: narrow};
    var svgWidth = layoutWidth + extraWidth + paddingLeft;
    var svgHeight = layoutHeight + paddingY * 2;
    [title.tails.svg, title.weave.front].forEach(function (svg) {
      svg.setAttribute('viewBox', (-paddingLeft).toFixed(2) + ' 0 ' + svgWidth.toFixed(2) + ' ' + svgHeight.toFixed(2));
      svg.style.width = svgWidth.toFixed(2) + 'px';
      svg.style.height = svgHeight.toFixed(2) + 'px';
      svg.style.left = (-paddingLeft).toFixed(2) + 'px';
      svg.style.top = (-paddingY).toFixed(2) + 'px';
    });
    title.tails.veil.setAttribute('x1', (-paddingLeft).toFixed(2));
    title.tails.veil.setAttribute('x2', (layoutWidth + fontSize * 1.5).toFixed(2));
    // The negative viewBox opens space before H without moving its native text origin.
    // 负 viewBox 为 H 前方留出空间；遮罩与画布共同扩边，原生文字坐标不发生位移。
    [title.tails.veilRect, title.tails.svg.querySelector('mask')].forEach(function (element) {
      element.setAttribute('x', (-paddingLeft).toFixed(2));
      element.setAttribute('y', '0');
      element.setAttribute('width', svgWidth.toFixed(2));
      element.setAttribute('height', svgHeight.toFixed(2));
    });
    title.element.dataset.codeReady = 'true';
  }

  // Soft background surfaces need fewer pixels than the expanded native SVG plume.
  // Keep their explicit paint budget small enough to leave time for the glyph flow.
  // 柔软背景曲面采用较低像素密度，显式限制绘制预算，为扩大后的原生 SVG 尾焰及文字流动留出时间。
  var desktopPixelRatio = 0.55;
  var mobilePixelRatio = 0.8;
  var maximumCanvasWidth = 1800;
  var maximumCanvasHeight = 1050;
  var maximumCanvasPixels = 220000;
  var pointerEase = 4;
  // Five visible scales illustrate HCCE; thin or distant folds use fewer scales.
  // 用五个可见尺度表达 HCCE；较远或较细的折面少用细尺度，避免纹理拥挤。
  var startingEncodingLevels = [0, 1, 2, 1, 3, 2];
  var maximumEncodingLevels = [2, 3, 3, 3, 4, 3];
  var minimumEncodingDuration = 7;
  var encodingDurationVariation = 6;
  var vertexSource = [
    'attribute vec2 a_position;',
    'varying vec2 v_uv;',
    'void main() {',
    '  v_uv = a_position * 0.5 + 0.5;',
    '  gl_Position = vec4(a_position, 0.0, 1.0);',
    '}'
  ].join('\n');

  // Thin folded surfaces form two flowing families, with dark space between them.
  // 轻薄折面分为两组柔性流向，层间保留暗部空隙，不堆成厚重色块。
  var fragmentSource = `
    #ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
    #else
    precision mediump float;
    #endif
    varying vec2 v_uv;
    uniform float u_aspect;
    uniform vec2 u_pointer;
    uniform float u_time;
    uniform float u_encoding_levels[6];

    // Level zero is the coordinate; later levels fold at successively doubled scales.
    // 第零层保留坐标，后续各层按二倍频率连续折叠；这是编码的视觉近似，非网络输出。
    vec3 coordinateCode(vec3 coordinates, float level) {
      if (level < 0.5) return coordinates;
      float frequency = exp2(level - 1.0);
      return 1.0 - abs(fract(coordinates * frequency) * 2.0 - 1.0);
    }

    vec3 codeColor(vec3 code) {
      code = smoothstep(vec3(0.02), vec3(0.98), code);
      float luminance = dot(code, vec3(0.2126, 0.7152, 0.0722));
      return clamp(mix(vec3(luminance), code, 1.28), 0.0, 1.0);
    }

    vec4 silk(vec2 p, float time, float phase, float tilt,
              float offset, float breadth, float depth, float encodingLevel, vec3 cool, vec3 warm) {
      float x = p.x;
      float center = offset + tilt * (x - 0.5);
      center += 0.18 * sin(x * 5.1 + time * 0.19 + phase);
      center += 0.025 * sin(x * 8.3 - time * 0.12 + phase * 1.7);
      float width = breadth * (0.94 + 0.16 * sin(x * 4.2 + time * 0.15 + phase));
      float across = (p.y - center) / width;
      float inside = 1.0 - smoothstep(0.90, 1.04, abs(across));
      // Thin folds leave most pixels empty; skip their color and lighting work.
      // 薄折面外的大部分像素完全透明，提前跳过色场与光照计算，增加层数也保持轻量。
      if (inside <= 0.0) return vec4(0.0);
      float roundness = sqrt(max(0.0, 1.0 - across * across));
      float foldPosition = 0.12 + 0.15 * sin(x * 3.7 - time * 0.16 + phase);
      float foldDistance = (across - foldPosition) * 15.0;
      float fold = exp(-foldDistance * foldDistance);
      // A lit face turns into a shaded face at the ridge, rather than a round tube.
      // 脊线两侧分别受光与背光，呈现薄片折叠的体积，避免圆鼓模糊的光管。
      float face = mix(0.15, 0.94, smoothstep(-1.0, foldPosition, across));
      face *= 1.0 - 0.72 * smoothstep(foldPosition, 1.02, across);
      float turned = smoothstep(foldPosition + 0.02, foldPosition + 0.16, across);
      turned *= 1.0 - smoothstep(foldPosition + 0.16, foldPosition + 0.46, across);
      float shade = face * (1.0 - 0.28 * turned) * (0.78 + depth * 0.48);
      float iridescence = 0.5 + 0.5 * sin(x * 2.6 + across * 1.65 - time * 0.14 + phase);
      vec3 tint = mix(cool, warm, iridescence);
      // Keep the same coordinate field through every level: regions subdivide in place.
      // 各层始终编码同一组坐标，颜色分区原位细分，不随机挪动坐标原点。
      vec3 coordinates = vec3(x * 0.72 + 0.13, across * 0.36 + 0.5,
        0.5 + 0.22 * sin(x * 3.2 + phase) + 0.18 * roundness);
      float coarseLevel = floor(encodingLevel);
      float nextLevel = min(coarseLevel + 1.0, 4.0);
      float blend = smoothstep(0.0, 1.0, fract(encodingLevel));
      vec3 coarseColor = codeColor(coordinateCode(coordinates, coarseLevel));
      vec3 nextColor = codeColor(coordinateCode(coordinates, nextLevel));
      // Crossfade adjacent encoded fields, not their frequencies; partitions stay nested.
      // 混合相邻层的编码色场，不连续拉伸频率；新分区从原有分区中平滑显现。
      // Color mapping precedes blending to avoid brightness jumps at the midpoint.
      // 先映射颜色再做融合，避免过渡中点的重新归一化造成亮度突变。
      vec3 coordinateColor = mix(coarseColor, nextColor, blend);
      tint = mix(tint, vec3(0.03) + coordinateColor * 0.94, 0.72 + depth * 0.08);
      // Keep the sheen chromatic so bright foreground folds stay clear rather than grey.
      // 高光保留青蓝与紫红色泽，让提亮后的近景褶皱清透而不蒙灰。
      vec3 pearl = mix(vec3(0.40, 0.86, 1.0), vec3(1.0, 0.48, 0.91), iridescence);
      vec3 color = tint * shade + pearl * fold * (0.16 + depth * 0.21);
      // Broad shaded rims give the surface volume instead of tracing hard neon lines.
      // 宽柔的暗边与褶皱赋予曲面体积，不沿轮廓描绘生硬的霓虹细线。
      float rim = smoothstep(0.40, 0.86, abs(across));
      color *= 1.0 - rim * (0.16 + depth * 0.04);
      float opacity = inside * (0.30 + depth * 0.43 + 0.08 * roundness);
      opacity *= smoothstep(0.08, 0.56, x);
      return vec4(color, opacity);
    }

    void main() {
      // Cover a 2.5:1 reference composition with equal physical scale on both axes.
      // 以 2.5:1 构图等比例覆盖：宽横幅裁上下，手机裁左右，不能纵向压扁图案。
      vec2 viewScale = vec2(min(u_aspect / 2.5, 1.0), min(2.5 / u_aspect, 1.0));
      vec2 viewCenter = vec2(mix(0.66, 0.5, viewScale.x), 0.5);
      vec2 p = viewCenter + (v_uv - 0.5) * viewScale;
      p += u_pointer * vec2(0.012, 0.018);
      float time = u_time;
      p.x += 0.025 * sin(p.y * 4.6 + time * 0.13);
      p.y += 0.025 * sin(p.x * 3.8 - time * 0.11);
      vec2 drift = vec2(sin(time * 0.09), cos(time * 0.11)) * 0.06;
      float haze = exp(-3.8 * length((p - vec2(0.73, 0.49) - drift) * vec2(1.0, 1.4)));
      vec3 color = mix(vec3(0.014, 0.034, 0.082), vec3(0.075, 0.15, 0.27), haze);
      color += vec3(0.11, 0.035, 0.19) * exp(-6.0 * length(p - vec2(0.93, 0.67)));

      // Uneven spacing and widths extend the right-hand folds into the middle.
      // 不等间距和宽度将右侧细腻的折面延展至中央，保留两组之间的深色留白。
      vec4 back = silk(p, time * 0.83, 2.2, 0.40, 0.68, 0.105, 0.08, u_encoding_levels[0],
        vec3(0.16, 0.31, 0.78), vec3(0.60, 0.24, 0.81));
      color = mix(color, back.rgb, back.a);
      vec4 upper = silk(p, time * 0.96, 4.5, 0.24, 0.62, 0.074, 0.36, u_encoding_levels[1],
        vec3(0.10, 0.52, 0.86), vec3(0.34, 0.64, 0.94));
      color = mix(color * (1.0 - upper.a * 0.06), upper.rgb, upper.a);
      vec4 middle = silk(p, time * 1.02, 4.9, 0.25, 0.49, 0.092, 0.58, u_encoding_levels[2],
        vec3(0.08, 0.60, 0.78), vec3(0.42, 0.45, 0.90));
      // Ordered translucent coverage preserves depth where surfaces cross.
      // 按前后顺序做半透明遮挡，交叠处保留层次，不通过滤色混成发白的杂色。
      color = mix(color * (1.0 - middle.a * 0.10), middle.rgb, middle.a);
      vec4 lower = silk(p, time * 0.94, 0.15, 0.63, 0.45, 0.057, 0.70, u_encoding_levels[3],
        vec3(0.12, 0.52, 0.86), vec3(0.53, 0.31, 0.89));
      color = mix(color * (1.0 - lower.a * 0.08), lower.rgb, lower.a);
      vec4 front = silk(p, time, 0.4, 0.66, 0.31, 0.096, 1.0, u_encoding_levels[4],
        vec3(0.23, 0.39, 0.91), vec3(0.78, 0.35, 0.85));
      color = mix(color * (1.0 - front.a * 0.14), front.rgb, front.a);
      vec4 companion = silk(p, time * 1.04, 0.66, 0.72, 0.20, 0.052, 0.78, u_encoding_levels[5],
        vec3(0.16, 0.56, 0.90), vec3(0.73, 0.25, 0.68));
      color = mix(color * (1.0 - companion.a * 0.09), companion.rgb, companion.a);
      color *= 0.55 + 0.45 * smoothstep(0.0, 0.7, p.x);
      gl_FragColor = vec4(color, 1.0);
    }
  `;

  var states = containers.map(function (container, index) {
    var bounds = container.getBoundingClientRect();
    var encodingTransitions = startingEncodingLevels.map(function (level, index) {
      var target = Math.floor(Math.random() * maximumEncodingLevels[index]);
      if (target >= level) target += 1;
      return {
        from: level,
        to: target,
        startedAt: -index * 0.8,
        duration: minimumEncodingDuration + Math.random() * encodingDurationVariation
      };
    });
    return {
      container: container,
      canvas: container.querySelector('.pose-flow-canvas'),
      image: container.querySelector('.pose-motion-art'),
      title: prepareTitle(container, index),
      gl: null,
      program: null,
      buffer: null,
      uniforms: null,
      ready: false,
      failed: false,
      contextLost: false,
      inViewport: bounds.bottom > 0 && bounds.top < window.innerHeight &&
        bounds.right > 0 && bounds.left < window.innerWidth,
      frame: null,
      previousTimestamp: null,
      elapsed: 0,
      encodingTransitions: encodingTransitions,
      encodingLevels: new Float32Array(startingEncodingLevels),
      pointerX: 0,
      pointerY: 0,
      targetPointerX: 0,
      targetPointerY: 0
    };
  });

  function stopRendering(state) {
    if (state.frame !== null) window.cancelAnimationFrame(state.frame);
    state.frame = null;
    state.previousTimestamp = null;
    state.pointerX = 0;
    state.pointerY = 0;
    state.targetPointerX = 0;
    state.targetPointerY = 0;
  }

  function releaseResources(state) {
    if (!state.gl || state.contextLost) return;
    if (state.buffer) state.gl.deleteBuffer(state.buffer);
    if (state.program) state.gl.deleteProgram(state.program);
    state.buffer = null;
    state.program = null;
  }

  function fallBackToImage(state) {
    stopRendering(state);
    state.ready = false;
    state.failed = true;
    delete state.container.dataset.flowReady;
    state.container.dataset.motion = 'paused';
    releaseResources(state);
  }

  function compileShader(gl, type, source) {
    var shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
    gl.deleteShader(shader);
    return null;
  }

  function resizeCanvas(state) {
    measureTitle(state.title);
    if (!state.ready || state.contextLost) return;
    var bounds = state.container.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    var pixelRatioCap = window.innerWidth <= 760 ? mobilePixelRatio : desktopPixelRatio;
    var scale = Math.min(window.devicePixelRatio || 1, pixelRatioCap,
      maximumCanvasWidth / bounds.width, maximumCanvasHeight / bounds.height,
      Math.sqrt(maximumCanvasPixels / (bounds.width * bounds.height)));
    var width = Math.max(2, Math.round(bounds.width * scale));
    var height = Math.max(2, Math.round(bounds.height * scale));
    if (state.canvas.width !== width || state.canvas.height !== height) {
      state.canvas.width = width;
      state.canvas.height = height;
    }
    var gl = state.gl;
    gl.viewport(0, 0, width, height);
    gl.useProgram(state.program);
    gl.uniform1f(state.uniforms.aspect, bounds.width / bounds.height);
  }

  function renderFrame(state, timestamp) {
    state.frame = null;
    if (!state.ready || state.contextLost || state.container.dataset.motion !== 'running') return;
    // Accumulate active time only; resuming never jumps forward through hidden time.
    // 时间只累计可见期间，恢复页面不会跨越隐藏时长而突然跳帧。
    // Use actual frame intervals so slower devices do not change the motion tempo.
    // 使用真实帧间隔，避免低帧率设备上的背景节奏变慢。
    var delta = state.previousTimestamp === null ? 0 : (timestamp - state.previousTimestamp) / 1000;
    state.previousTimestamp = timestamp;
    state.elapsed += delta;
    var pointerBlend = 1 - Math.exp(-pointerEase * delta);
    state.pointerX += (state.targetPointerX - state.pointerX) * pointerBlend;
    state.pointerY += (state.targetPointerY - state.pointerY) * pointerBlend;
    // Random targets use active time and survive visibility changes and GL restoration.
    // 随机目标只按可见时间推进；离屏、切换系统动效偏好或恢复 GL 时保留当前阶段。
    state.encodingTransitions.forEach(function (transition, index) {
      if (state.elapsed >= transition.startedAt + transition.duration) {
        transition.from = transition.to;
        var target = Math.floor(Math.random() * maximumEncodingLevels[index]);
        if (target >= transition.from) target += 1;
        transition.to = target;
        transition.startedAt = state.elapsed;
        transition.duration = minimumEncodingDuration + Math.random() * encodingDurationVariation;
      }
      var progress = Math.max(0, Math.min(1, (state.elapsed - transition.startedAt) / transition.duration));
      // Quintic easing has zero velocity and acceleration at each random changeover.
      // 五次缓动使随机段落衔接处的速度、加速度均归零，随后依次经过相邻尺度。
      var blend = progress * progress * progress * (progress * (progress * 6 - 15) + 10);
      state.encodingLevels[index] = transition.from + (transition.to - transition.from) * blend;
    });
    var gl = state.gl;
    gl.uniform1f(state.uniforms.time, state.elapsed);
    gl.uniform2f(state.uniforms.pointer, state.pointerX, state.pointerY);
    gl.uniform1fv(state.uniforms.encodingLevels, state.encodingLevels);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (state.container.dataset.flowReady !== 'true') state.container.dataset.flowReady = 'true';
    state.frame = window.requestAnimationFrame(function (nextTimestamp) {
      renderFrame(state, nextTimestamp);
    });
  }

  function updateActivity(state) {
    var permitted = !reducedMotion.matches && !forcedColors.matches;
    var visible = state.inViewport && document.visibilityState === 'visible';
    var running = permitted && visible && state.ready && !state.contextLost;
    // Native CSS/SVG title flow must not depend on the background's WebGL support.
    // 文字的原生 CSS/SVG 动效不依赖背景 WebGL 是否成功；离屏及系统减弱动效时仍统一暂停。
    var titleMotion = permitted && visible ? 'running' : 'paused';
    if (state.container.dataset.titleMotion !== titleMotion) {
      state.container.dataset.titleMotion = titleMotion;
      if (state.title) {
        [state.title.svg, state.title.tails.svg, state.title.weave.front].forEach(function (svg) {
          if (typeof svg.pauseAnimations !== 'function') return;
          if (titleMotion === 'running') svg.unpauseAnimations();
          else svg.pauseAnimations();
        });
      }
    }
    if (state.title) {
      var tails = state.title.tails;
      if (titleMotion === 'running' && tails.frame === null) {
        tails.frame = window.requestAnimationFrame(function (timestamp) { renderTitleTails(state.title, timestamp); });
      } else if (titleMotion !== 'running') {
        if (tails.frame !== null) window.cancelAnimationFrame(tails.frame);
        tails.frame = null;
        tails.previousTimestamp = null;
      }
    }
    state.container.dataset.motionVisible = String(visible);
    state.container.dataset.motion = running ? 'running' : 'paused';
    state.container.dataset.motionReady = 'true';
    if (!permitted) delete state.container.dataset.flowReady;
    if (!running) {
      stopRendering(state);
      return;
    }
    if (state.frame === null) {
      state.frame = window.requestAnimationFrame(function (timestamp) {
        renderFrame(state, timestamp);
      });
    }
  }

  function initializeRenderer(state) {
    if (!state.canvas || !state.image || state.ready || state.failed || state.contextLost ||
        reducedMotion.matches || forcedColors.matches) return;
    try {
      var gl = state.canvas.getContext('webgl', {
        alpha: false, antialias: false, depth: false, stencil: false,
        preserveDrawingBuffer: false, powerPreference: 'low-power'
      });
      if (!gl) {
        fallBackToImage(state);
        return;
      }
      state.gl = gl;
      var vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
      var fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
      if (!vertexShader || !fragmentShader) {
        if (vertexShader) gl.deleteShader(vertexShader);
        if (fragmentShader) gl.deleteShader(fragmentShader);
        fallBackToImage(state);
        return;
      }
      state.program = gl.createProgram();
      gl.attachShader(state.program, vertexShader);
      gl.attachShader(state.program, fragmentShader);
      gl.linkProgram(state.program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      if (!gl.getProgramParameter(state.program, gl.LINK_STATUS)) {
        fallBackToImage(state);
        return;
      }
      gl.useProgram(state.program);
      state.buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, state.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
      var position = gl.getAttribLocation(state.program, 'a_position');
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      state.uniforms = {
        time: gl.getUniformLocation(state.program, 'u_time'),
        pointer: gl.getUniformLocation(state.program, 'u_pointer'),
        aspect: gl.getUniformLocation(state.program, 'u_aspect'),
        encodingLevels: gl.getUniformLocation(state.program, 'u_encoding_levels[0]')
      };
      state.ready = true;
      resizeCanvas(state);
      updateActivity(state);
    } catch (error) {
      // The original image remains usable when WebGL is blocked.
      // WebGL 受限时，继续显示原始背景图。
      fallBackToImage(state);
    }
  }

  states.forEach(function (state) {
    measureTitle(state.title);
    if (!state.canvas || !state.image) return;
    state.canvas.addEventListener('webglcontextlost', function (event) {
      event.preventDefault();
      state.contextLost = true;
      state.ready = false;
      state.failed = false;
      stopRendering(state);
      delete state.container.dataset.flowReady;
      updateActivity(state);
    });
    state.canvas.addEventListener('webglcontextrestored', function () {
      state.contextLost = false;
      state.program = null;
      state.buffer = null;
      initializeRenderer(state);
    });
    state.container.addEventListener('pointermove', function (event) {
      if (!finePointer.matches || event.pointerType === 'touch' || state.container.dataset.motion !== 'running') return;
      var bounds = state.container.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      state.targetPointerX = Math.max(-1, Math.min(1, (event.clientX - bounds.left) / bounds.width * 2 - 1));
      state.targetPointerY = Math.max(-1, Math.min(1, 1 - (event.clientY - bounds.top) / bounds.height * 2));
    });
    state.container.addEventListener('pointerleave', function () {
      state.targetPointerX = 0;
      state.targetPointerY = 0;
    });
    state.container.addEventListener('pointercancel', function () {
      state.targetPointerX = 0;
      state.targetPointerY = 0;
    });
    initializeRenderer(state);
    updateActivity(state);
  });

  // Both observers only manage lifecycle and size; surface movement stays on the GPU.
  // 观察器仅处理可见性和尺寸；所有曲面变形都在 GPU 中完成。
  if ('IntersectionObserver' in window) {
    var visibilityObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var state = states.find(function (item) { return item.container === entry.target; });
        state.inViewport = entry.isIntersecting;
        updateActivity(state);
      });
    });
    states.forEach(function (state) { visibilityObserver.observe(state.container); });
  }
  if ('ResizeObserver' in window) {
    var sizeObserver = new ResizeObserver(function (entries) {
      entries.forEach(function (entry) {
        var state = states.find(function (item) { return item.container === entry.target; });
        resizeCanvas(state);
      });
    });
    states.forEach(function (state) { sizeObserver.observe(state.container); });
  }
  window.addEventListener('resize', function () { states.forEach(resizeCanvas); });
  if (document.fonts) document.fonts.ready.then(function () { states.forEach(function (state) { measureTitle(state.title); }); });
  document.addEventListener('visibilitychange', function () { states.forEach(updateActivity); });
  function updateSystemPreference() {
    states.forEach(function (state) {
      initializeRenderer(state);
      updateActivity(state);
    });
  }
  reducedMotion.addEventListener('change', updateSystemPreference);
  forcedColors.addEventListener('change', updateSystemPreference);
  finePointer.addEventListener('change', function () {
    states.forEach(function (state) {
      state.targetPointerX = 0;
      state.targetPointerY = 0;
    });
  });
})();
