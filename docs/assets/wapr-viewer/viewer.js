// Offline saved-scene inspection; this never runs pose estimation in the browser.
// 离线检查已有真实场景，浏览器不执行位姿推理；OpenCV 米制坐标转为 Three.js 的 y-up 视图。
import * as THREE from './three.module.min.js';
import {OrbitControls} from './OrbitControls.js?v=orbit-capture-release-20261118';
import {TransformControls} from './TransformControls.js';

const root = document.documentElement;
const compact = new URLSearchParams(location.search).get('compact') === '1';
root.dataset.compact = String(compact);
const photo = document.getElementById('photo');
const canvas = document.getElementById('cloud');
const loading = document.getElementById('loading');
const selection = document.getElementById('selection');
const instanceButtons = document.getElementById('instances');
const meshToggle = document.getElementById('mesh-toggle');
const wireframeToggle = document.getElementById('edges-toggle');
const passColor = 0x53dfb1;
const failColor = 0xff8997;
const duplicateColor = 0xf0bd69;
let sceneData, renderer, scene, camera, controls, cloud, meshes = [];
let selectedId = null;
let mode = compact ? 'accepted' : 'all';
let image;
let viewPivot, displayRoot, manipulator, rotationTicks, viewRingTicks;
let axisLabels = [];
let viewTool = compact ? 'orbit' : 'translate';
let gizmoPointer = null;
const viewPointers = new Map();
const orbitPointers = new Set();
let pickGesture = null;
let replayingOrbit = false;

function language() { return root.dataset.lang === 'zh' ? 'zh' : 'en'; }

// Keep live readings next to the 3D canvas when the RGB pane stacks above it.
// 单列布局中把实时读数移到右画布旁，避免中间隔着整幅RGB图而看不到反馈。
const viewNavigation = document.querySelector('.view-navigation');
const views = document.querySelector('.views');
const singleColumn = window.matchMedia('(max-width: 620px)');
function placeViewNavigation() {
  if (singleColumn.matches) canvas.closest('.view').insertBefore(viewNavigation, canvas.parentElement);
  else views.parentElement.insertBefore(viewNavigation, views);
  resizeFrame();
}
singleColumn.addEventListener('change', placeViewNavigation);
placeViewNavigation();

function resizeFrame() {
  // Restore both dimensions even when the parent has a cached stylesheet without this viewer's rules.
  // 父页面可能缓存了没有本查看器规则的旧样式，因此宽、高都显式设置，避免默认 300px 窄框。
  const frame = window.frameElement;
  if (!frame || compact) return;
  frame.style.display = 'block';
  frame.style.width = '100%';
  frame.style.maxWidth = '100%';
  frame.style.boxSizing = 'border-box';
  frame.style.border = '0';
  frame.style.height = Math.ceil(document.body.scrollHeight + 2) + 'px';
}

function syncPreferences() {
  const requestedLanguage = new URLSearchParams(location.search).get('lang');
  let savedLanguage = null;
  try { savedLanguage = localStorage.getItem('hccepose-language'); } catch (error) { /* Storage can be blocked. / 存储可能被禁用。 */ }
  const preferredLanguage = requestedLanguage === 'en' || requestedLanguage === 'zh'
    ? requestedLanguage : savedLanguage === 'zh' ? 'zh' : 'en';
  try {
    const parentRoot = window.parent.document.documentElement;
    if (window.parent === window) {
      root.dataset.lang = preferredLanguage;
      root.dataset.theme = localStorage.getItem('hccepose-theme') === 'dark' ? 'dark' : 'light';
    } else {
      const parentLanguage = parentRoot.dataset.lang;
      root.dataset.lang = parentLanguage === 'en' || parentLanguage === 'zh' ? parentLanguage : preferredLanguage;
      root.dataset.theme = parentRoot.dataset.theme || 'light';
    }
  } catch (error) {
    root.dataset.lang = preferredLanguage;
  }
  root.lang = language() === 'en' ? 'en' : 'zh-CN';
  for (const element of document.querySelectorAll('[data-label-zh]')) {
    element.setAttribute('aria-label', language() === 'en' ? element.dataset.labelEn : element.dataset.labelZh);
  }
  for (const element of document.querySelectorAll('[data-title-zh]')) {
    element.title = language() === 'en' ? element.dataset.titleEn : element.dataset.titleZh;
  }
  for (const label of instanceButtons.querySelectorAll('.instance-label')) {
    label.textContent = (language() === 'zh' ? '框 ' : 'Box ') + label.parentElement.dataset.id;
  }
  updateSelection(); drawPhoto();
  updateNavigationHint();
  resizeFrame();
}

function visible(row) { return mode === 'all' || (mode === 'accepted' && row.accepted) || (mode === 'selected' && row.id === selectedId); }

function updateSelection() {
  if (root.dataset.ready === 'error') {
    const message = language() === 'zh'
      ? '三维视图加载失败。请刷新页面，并检查场景资源及浏览器的 WebGL 支持。'
      : 'The 3D view could not load. Reload the page and check the scene resources and WebGL support.';
    loading.textContent = message;
    selection.textContent = message;
    return;
  }
  if (!sceneData) return;
  const row = sceneData.instances.find(item => item.id === selectedId);
  if (!row) return;
  const duplicate = row.status === 'suppressed';
  const status = language() === 'zh'
    ? (row.accepted ? '已保留' : (duplicate ? '重复，已保留框 ' + row.nms.suppressed_by_detection_id : '低于阈值'))
    : (row.accepted ? 'Accepted' : (duplicate ? 'Duplicate of Box ' + row.nms.suppressed_by_detection_id : 'Below threshold'));
  selection.textContent = '#' + row.rank + (language() === 'zh' ? ' · 框 ' : ' · Box ') + row.id + ' · WBPS ' + row.score.toFixed(6) + ' · ' + status;
}

function drawPhoto() {
  if (!sceneData || !image || !image.complete || !image.naturalWidth) return;
  const ctx = photo.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const rows = sceneData.instances.filter(visible);
  // Draw the selected instance last; only saved projected 3D edges are shown, not fabricated masks.
  // 选中实例最后绘制；只使用已保存的 3D 投影边，不伪造预测掩码或真值。
  rows.sort((a, b) => Number(a.id === selectedId) - Number(b.id === selectedId));
  for (const row of rows) {
    const active = row.id === selectedId;
    ctx.strokeStyle = row.accepted ? '#53dfb1' : (row.status === 'suppressed' ? '#f0bd69' : '#ff8997');
    ctx.lineWidth = active ? 3 : 1.2;
    ctx.globalAlpha = active ? 1 : 0.45;
    ctx.beginPath();
    for (const edge of row.edges) { ctx.moveTo(...edge[0]); ctx.lineTo(...edge[1]); }
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (active) {
      const text = (language() === 'zh' ? '框 ' : 'Box ') + row.id + '  ' + row.score.toFixed(6);
      ctx.font = 'bold 20px sans-serif';
      const textWidth = ctx.measureText(text).width + 16;
      const x = Math.min(photo.width - textWidth, Math.max(0, row.box[0]));
      const y = Math.max(28, row.box[1] - 8);
      ctx.fillStyle = '#071525e8'; ctx.fillRect(x, y - 25, textWidth, 31);
      ctx.fillStyle = row.accepted ? '#9ff3d7' : (row.status === 'suppressed' ? '#ffe1ad' : '#ffbdc6'); ctx.fillText(text, x + 8, y);
    }
  }
}

function pickProjectedInstance(x, y) {
  // Pick inside the drawn 3D box's projected hull; overlap resolves to the nearest drawn edge.
  // 按已画出的三维框投影凸包拾取；重叠时选择距点击位置最近的投影边，不使用未画出的检测框。
  const ctx = photo.getContext('2d');
  let closest = null;
  let closestDistanceSquared = Infinity;
  for (const row of sceneData.instances.filter(visible)) {
    const corners = row.edges.flat().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const points = corners.filter((point, i) => i === 0 || point[0] !== corners[i - 1][0] || point[1] !== corners[i - 1][1]);
    const hull = [];
    for (const ordered of [points, points.slice().reverse()]) {
      const half = [];
      for (const point of ordered) {
        while (half.length >= 2) {
          const a = half[half.length - 2];
          const b = half[half.length - 1];
          const cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
          if (cross > 0) break;
          half.pop();
        }
        half.push(point);
      }
      half.pop();
      hull.push(...half);
    }
    if (hull.length < 3) continue;
    const outline = new Path2D();
    outline.moveTo(...hull[0]);
    for (const point of hull.slice(1)) outline.lineTo(...point);
    outline.closePath();
    if (!ctx.isPointInPath(outline, x, y)) continue;
    let distanceSquared = Infinity;
    for (const [a, b] of row.edges) {
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const lengthSquared = dx * dx + dy * dy;
      const t = lengthSquared === 0 ? 0 : Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / lengthSquared));
      const edgeDistanceSquared = (x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2;
      distanceSquared = Math.min(distanceSquared, edgeDistanceSquared);
    }
    if (distanceSquared < closestDistanceSquared) {
      closest = row;
      closestDistanceSquared = distanceSquared;
    }
  }
  return closest;
}

function updateNavigationHint() {
  const hint = document.getElementById('view-help');
  if (!hint) return;
  const hints = language() === 'zh'
    ? {translate: '拖动彩色轴平移 · 滚轮缩放', rotate: '拖动刻度环 · 每格5° · 滚轮缩放', orbit: '拖动旋转 · 右键/Shift 拖动平移 · 滚轮缩放'}
    : {translate: 'Drag axes to pan · Scroll to zoom', rotate: 'Drag ring · 5° ticks · Scroll to zoom', orbit: 'Drag to orbit · Right/Shift-drag to pan · Scroll to zoom'};
  // Touch users need the gesture hint instead of a mouse-wheel instruction.
  // 触摸设备显示手势说明，不把手机用户引向并不存在的鼠标滚轮。
  if (window.matchMedia('(pointer: coarse)').matches) {
    if (language() === 'zh') {
      hints.translate = '拖动彩色轴 · 双指缩放';
      hints.rotate = '拖动刻度环 · 每格5° · 双指缩放';
      hints.orbit = '单指旋转 · 双指平移/缩放';
    } else {
      hints.translate = 'Drag axes · Pinch to zoom';
      hints.rotate = 'Drag ring · 5° ticks · Pinch to zoom';
      hints.orbit = 'One finger: orbit · Two: pan/zoom';
    }
  }
  hint.textContent = hints[viewTool];
  const feedback = document.getElementById('view-feedback');
  if (manipulator && manipulator.dragging) {
    const axis = manipulator.axis || '';
    if (viewTool === 'rotate') {
      feedback.textContent = axis + '  ' + (manipulator.rotationAngle * 180 / Math.PI).toFixed(1) + '°';
    } else {
      const delta = viewPivot.position.clone().sub(manipulator.worldPositionStart);
      const component = ['X', 'Y', 'Z'].indexOf(axis);
      const distance = component === -1 ? delta.length() : delta.getComponent(component);
      feedback.textContent = axis + '  ' + (distance * 1000).toFixed(1) + ' mm';
    }
  } else if (manipulator && manipulator.axis && viewTool !== 'orbit') {
    feedback.textContent = manipulator.axis + ' · ' + (language() === 'zh'
      ? (viewTool === 'rotate' ? '拖动旋转' : '拖动平移')
      : (viewTool === 'rotate' ? 'Drag to rotate' : 'Drag to pan'));
  } else {
    feedback.textContent = '';
  }
}

function updateNavigationDecorations() {
  if (!manipulator) return;
  const active = viewTool !== 'orbit';
  const center = viewPivot.getWorldPosition(new THREE.Vector3());
  const eye = camera.position.clone().sub(center).normalize();
  // Match r160's screen-sized handles; these display dimensions never rescale the saved model.
  // 与r160手柄的屏幕尺寸一致；只缩放操作标记，不改变保存模型的物理尺寸。
  const size = center.distanceTo(camera.position)
    * Math.min(1.9 * Math.tan(Math.PI * camera.fov / 360) / camera.zoom, 7) * manipulator.size / 4;
  rotationTicks.visible = active && viewTool === 'rotate';
  rotationTicks.position.copy(center);
  rotationTicks.scale.setScalar(size);
  viewRingTicks.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), eye);
  for (const label of axisLabels) {
    label.visible = active && (viewTool === 'rotate' || Math.abs(label.userData.direction.dot(eye)) < 0.97);
    const direction = viewTool === 'rotate' ? label.userData.ringDirection : label.userData.direction;
    label.position.copy(center).addScaledVector(direction, size * 0.72);
    const worldUnitsPerPixel = 2 * center.distanceTo(camera.position) * Math.tan(Math.PI * camera.fov / 360) / Math.max(1, canvas.clientHeight);
    const labelSize = Math.max(size * 0.14, worldUnitsPerPixel * 20);
    label.scale.set(labelSize, labelSize, 1);
  }
  canvas.style.cursor = manipulator.dragging ? 'grabbing' : active && manipulator.axis ? 'grab' : 'default';
  updateNavigationHint();
}

function render() {
  if (!renderer || !camera) return;
  updateNavigationDecorations();
  renderer.render(scene, camera);
}

function placeViewPivot(resetDisplay = false) {
  if (!viewPivot || selectedId === null) return;
  const row = sceneData.instances.find(item => item.id === selectedId);
  const point = new THREE.Vector3(row.pose[3], -row.pose[7], -row.pose[11]);
  // A presentation-only parent moves cloud and CAD together; raw vertices and poses stay untouched.
  // 展示父节点同时移动点云和CAD，原始顶点和位姿不变；换选中项时保持右侧画面连续。
  scene.updateMatrixWorld(true);
  const previous = displayRoot.matrixWorld.clone();
  if (!resetDisplay) displayRoot.localToWorld(point);
  viewPivot.position.copy(point);
  viewPivot.quaternion.identity();
  viewPivot.scale.setScalar(1);
  viewPivot.updateMatrixWorld(true);
  if (resetDisplay) displayRoot.matrix.makeTranslation(-point.x, -point.y, -point.z);
  else displayRoot.matrix.copy(viewPivot.matrixWorld).invert().multiply(previous);
  displayRoot.updateMatrixWorld(true);
}

function createNavigationControls() {
  manipulator = new TransformControls(camera, canvas);
  // Route pointer IDs here instead of letting r160's ID-agnostic handlers share a touch drag.
  // 在查看器里按指针ID分配手势，避免r160不区分ID的原生手柄事件被第二指接管。
  canvas.removeEventListener('pointerdown', manipulator._onPointerDown);
  canvas.removeEventListener('pointermove', manipulator._onPointerHover);
  canvas.removeEventListener('pointerup', manipulator._onPointerUp);
  manipulator.setSpace('world');
  manipulator.setSize(1.35);
  scene.add(manipulator);
  rotationTicks = new THREE.Group();
  scene.add(rotationTicks);
  const viewMarks = [];
  for (let degrees = 0; degrees < 360; degrees += 5) {
    const angle = degrees * Math.PI / 180;
    for (const radius of [0.765, degrees % 30 === 0 ? 0.85 : 0.8]) {
      viewMarks.push(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
    }
  }
  viewRingTicks = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(viewMarks, 3)),
    new THREE.LineBasicMaterial({color: 0xb5d2e9, transparent: true, opacity: 0.85, depthTest: false, depthWrite: false})
  );
  viewRingTicks.renderOrder = Infinity;
  rotationTicks.add(viewRingTicks);
  const axes = [
    {name: 'X', color: '#ff6c7f', direction: new THREE.Vector3(1, 0, 0)},
    {name: 'Y', color: '#76e4a2', direction: new THREE.Vector3(0, 1, 0)},
    {name: 'Z', color: '#71b8ff', direction: new THREE.Vector3(0, 0, 1)}
  ];
  for (const axis of axes) {
    const points = [];
    // Five-degree marks and longer thirty-degree marks form a continuous, unsnapped protractor.
    // 每5度一格、30度长刻度；只提供角度参照，不强制吸附或限制自由旋转。
    for (let degrees = 0; degrees < 360; degrees += 5) {
      const angle = degrees * Math.PI / 180;
      for (const radius of [0.515, degrees % 30 === 0 ? 0.58 : 0.54]) {
        const a = Math.cos(angle) * radius;
        const b = Math.sin(angle) * radius;
        points.push(...(axis.name === 'X' ? [0, a, b] : axis.name === 'Y' ? [a, 0, b] : [a, b, 0]));
      }
    }
    const ticks = new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3)),
      new THREE.LineBasicMaterial({color: axis.color, transparent: true, opacity: 0.8, depthTest: false, depthWrite: false})
    );
    ticks.renderOrder = Infinity;
    rotationTicks.add(ticks);
    const badge = document.createElement('canvas');
    badge.width = 64; badge.height = 64;
    const context = badge.getContext('2d');
    context.fillStyle = '#091729'; context.beginPath(); context.arc(32, 32, 27, 0, Math.PI * 2); context.fill();
    context.strokeStyle = axis.color; context.lineWidth = 3; context.stroke();
    context.fillStyle = axis.color; context.font = 'bold 38px sans-serif'; context.textAlign = 'center'; context.textBaseline = 'middle';
    context.fillText(axis.name, 32, 34);
    const label = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(badge), depthTest: false, depthWrite: false, toneMapped: false}));
    label.userData.direction = axis.direction;
    label.userData.ringDirection = axis.name === 'X' ? new THREE.Vector3(0, 1, 0)
      : axis.name === 'Y' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(-Math.SQRT1_2, -Math.SQRT1_2, 0);
    label.renderOrder = Infinity; scene.add(label); axisLabels.push(label);
  }
  manipulator.addEventListener('change', render);
  manipulator.addEventListener('dragging-changed', event => {
    controls.enabled = !event.value;
    for (const button of document.querySelectorAll('[data-view-tool]')) button.disabled = event.value;
    render();
  });
  for (const button of document.querySelectorAll('[data-view-tool]')) button.addEventListener('click', () => {
    viewTool = button.dataset.viewTool;
    if (viewTool === 'orbit') manipulator.detach();
    else { manipulator.setMode(viewTool); manipulator.attach(viewPivot); }
    for (const item of document.querySelectorAll('[data-view-tool]')) item.setAttribute('aria-pressed', String(item === button));
    render();
  });
  if (viewTool !== 'orbit') { manipulator.setMode(viewTool); manipulator.attach(viewPivot); }
  else manipulator.detach();
  for (const button of document.querySelectorAll('[data-view-tool]')) button.disabled = false;
  bindCanvasGestures();
}

function finishGizmoDrag(releaseCapture = true) {
  const pointerId = gizmoPointer;
  gizmoPointer = null;
  if (manipulator) {
    if (manipulator.dragging) manipulator.pointerUp({button: 0});
    manipulator.axis = null;
  }
  controls.enabled = true;
  if (releaseCapture && pointerId !== null && canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
}

function forwardOrbitPointer(type, pointer) {
  // Re-seed OrbitControls from current touches at a handoff, without re-running our picking logic.
  // 切换到双指时用当前触点重建OrbitControls状态，不再次触发本层拾取或拖轴。
  replayingOrbit = true;
  try {
    if (type === 'pointerdown') orbitPointers.add(pointer.pointerId);
    else orbitPointers.delete(pointer.pointerId);
    canvas.dispatchEvent(new PointerEvent(type, {bubbles: true, view: window,
      pointerId: pointer.pointerId, pointerType: pointer.pointerType, isPrimary: pointer.isPrimary,
      clientX: pointer.clientX, clientY: pointer.clientY, button: 0, buttons: type === 'pointerdown' ? 1 : 0}));
  } finally {
    replayingOrbit = false;
  }
}

function restartOrbitTouches() {
  const touches = Array.from(viewPointers.values()).filter(pointer => pointer.pointerType === 'touch');
  if (gizmoPointer !== null || touches.length === 0) return;
  // r160 ends its gesture whenever either finger lifts. Rebase survivors before the next movement.
  // r160在任一手指抬起时结束当前手势；用剩余触点重新起步，防止停住或下一次移动跳变。
  for (const pointer of touches) if (orbitPointers.has(pointer.pointerId)) forwardOrbitPointer('pointercancel', pointer);
  for (const pointer of touches) forwardOrbitPointer('pointerdown', pointer);
}

function cancelCanvasGesture() {
  const pointers = Array.from(viewPointers.values());
  viewPointers.clear(); pickGesture = null;
  finishGizmoDrag();
  for (const pointer of pointers) if (orbitPointers.has(pointer.pointerId)) forwardOrbitPointer('pointercancel', pointer);
}

function cancelTrackedPointer(pointerId) {
  if (pickGesture) pickGesture.cancelled = true;
  const pointer = viewPointers.get(pointerId);
  viewPointers.delete(pointerId);
  if (gizmoPointer === pointerId) finishGizmoDrag(false);
  else if (orbitPointers.has(pointerId)) forwardOrbitPointer('pointercancel', pointer);
  if (viewPointers.size === 0) pickGesture = null;
  else setTimeout(restartOrbitTouches, 0);
}

function bindCanvasGestures() {
  const raycaster = new THREE.Raycaster();
  function handlePointer(event) {
    const rect = canvas.getBoundingClientRect();
    return {x: (event.clientX - rect.left) / rect.width * 2 - 1,
      y: 1 - (event.clientY - rect.top) / rect.height * 2, button: event.button};
  }
  canvas.addEventListener('pointerdown', event => {
    if (replayingOrbit) return;
    if (viewPointers.size === 0) pickGesture = {pointerId: event.pointerId, x: event.clientX, y: event.clientY,
      cancelled: event.button !== 0 || !event.isPrimary || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey};
    else if (pickGesture) pickGesture.cancelled = true;
    viewPointers.set(event.pointerId, event);
    if (viewPointers.size > 1 && gizmoPointer !== null) {
      const first = viewPointers.get(gizmoPointer);
      finishGizmoDrag(false);
      forwardOrbitPointer('pointerdown', first);
    }
    if (viewPointers.size === 1 && viewTool !== 'orbit' && event.button === 0) {
      manipulator.pointerHover(handlePointer(event));
      if (manipulator.axis) {
        pickGesture.cancelled = true;
        gizmoPointer = event.pointerId;
        controls.enabled = false;
        canvas.setPointerCapture(event.pointerId);
        manipulator.pointerDown(handlePointer(event));
        event.stopImmediatePropagation();
        return;
      }
    }
    orbitPointers.add(event.pointerId);
  }, {capture: true});
  canvas.addEventListener('pointermove', event => {
    if (replayingOrbit) return;
    if (viewPointers.has(event.pointerId)) {
      viewPointers.set(event.pointerId, event);
      const samples = event.getCoalescedEvents ? [...event.getCoalescedEvents(), event] : [event];
      if (pickGesture && pickGesture.pointerId === event.pointerId && (event.buttons !== 1 ||
        samples.some(sample => Math.hypot(sample.clientX - pickGesture.x, sample.clientY - pickGesture.y) > 5))) pickGesture.cancelled = true;
      if (gizmoPointer === event.pointerId) {
        manipulator.pointerMove(handlePointer(event));
        event.stopImmediatePropagation();
      }
    } else if (viewPointers.size > 0 || event.pointerType === 'touch') {
      event.stopImmediatePropagation();
    } else if (viewTool !== 'orbit') {
      manipulator.pointerHover(handlePointer(event));
    }
  }, {capture: true});
  canvas.addEventListener('pointerup', event => {
    if (replayingOrbit) return;
    if (!viewPointers.has(event.pointerId)) { event.stopImmediatePropagation(); return; }
    const shouldPick = pickGesture && pickGesture.pointerId === event.pointerId && !pickGesture.cancelled &&
      event.button === 0 && viewPointers.size === 1 && Math.hypot(event.clientX - pickGesture.x, event.clientY - pickGesture.y) <= 5;
    viewPointers.delete(event.pointerId); orbitPointers.delete(event.pointerId);
    if (gizmoPointer === event.pointerId) { finishGizmoDrag(); event.stopImmediatePropagation(); }
    if (viewPointers.size === 0) pickGesture = null;
    else if (event.pointerType === 'touch') setTimeout(restartOrbitTouches, 0);
    if (!shouldPick) return;
    raycaster.setFromCamera(handlePointer(event), camera);
    const hit = raycaster.intersectObjects(meshes.filter(mesh => mesh.visible), false)[0];
    if (hit) selectInstance(hit.object.userData.row.id);
  }, {capture: true});
  canvas.addEventListener('pointercancel', event => {
    if (replayingOrbit) return;
    if (!viewPointers.has(event.pointerId)) { event.stopImmediatePropagation(); return; }
    if (pickGesture) pickGesture.cancelled = true;
    viewPointers.delete(event.pointerId); orbitPointers.delete(event.pointerId);
    if (gizmoPointer === event.pointerId) { finishGizmoDrag(); event.stopImmediatePropagation(); }
    if (viewPointers.size === 0) pickGesture = null;
    else setTimeout(restartOrbitTouches, 0);
  }, {capture: true});
  canvas.addEventListener('lostpointercapture', event => {
    if (replayingOrbit || !viewPointers.has(event.pointerId)) return;
    cancelTrackedPointer(event.pointerId);
  });
  // Parent-page events cannot reach this document; cancel tracked touches when they leave it entirely.
  // 触点越过iframe后，父页事件不会传回本子文档；离开整个文档时立即取消，不影响文档内移动。
  root.addEventListener('pointerleave', event => {
    if (replayingOrbit || event.target !== root || !viewPointers.has(event.pointerId)) return;
    const nextTarget = event.relatedTarget;
    if (nextTarget && (nextTarget === document || nextTarget.ownerDocument === document)) return;
    cancelTrackedPointer(event.pointerId);
  });
  // Rebase a surviving touch at its latest position, even after it moves outside uncaptured.
  // 失去待生效捕获的触点移出画布后仍记录当前位置，余指接管时不把整段位移算入旋转。
  document.addEventListener('pointermove', event => {
    if (replayingOrbit || canvas.contains(event.target) || !viewPointers.has(event.pointerId)) return;
    viewPointers.set(event.pointerId, event);
    if (pickGesture) pickGesture.cancelled = true;
  }, {capture: true});
  // Withdrawing pending capture may emit no lostpointercapture. An outside release still ends the touch.
  // 尚未生效的捕获被撤销时可能没有lostpointercapture；画布外松手仍须清理并衔接剩余触点。
  for (const type of ['pointerup', 'pointercancel']) document.addEventListener(type, event => {
    if (replayingOrbit || canvas.contains(event.target) || !viewPointers.has(event.pointerId)) return;
    cancelTrackedPointer(event.pointerId);
  }, {capture: true});
}

function updateScene() {
  if (!sceneData) return;
  wireframeToggle.disabled = !meshToggle.checked;
  for (const mesh of meshes) {
    mesh.visible = meshToggle.checked && visible(mesh.userData.row);
    const active = mesh.userData.row.id === selectedId;
    // CAD surfaces stay distinct from neutral sensor points; selection adds a gold feature outline.
    // CAD 面与中性测量散点保持区分；选中时增加金色结构轮廓，不改变分数的绿/红含义。
    mesh.material.opacity = active ? 0.96 : 0.62;
    mesh.material.emissive.setHex(active ? (mesh.userData.row.accepted ? 0x113524 : 0x35151a) : 0);
    mesh.userData.outline.visible = active;
  }
  for (const button of instanceButtons.children) {
    button.disabled = mode === 'accepted' && button.dataset.pass !== 'true';
    button.setAttribute('aria-pressed', String(Number(button.dataset.id) === selectedId));
  }
  updateSelection(); drawPhoto(); render();
}

function selectInstance(id) {
  const row = sceneData.instances.find(item => item.id === id);
  if (!row || (mode === 'accepted' && !row.accepted)) return;
  selectedId = id;
  // Orbit the selected copy while keeping the current camera position.
  // 围绕选中的这份实例旋转，保留当前相机位置，不把同类多个实例混为一份。
  placeViewPivot();
  controls.target.copy(viewPivot.position);
  controls.update();
  updateScene();
}

function updateSensorProjection() {
  // Use the complete calibrated frustum, including off-center cx/cy and independent fx/fy.
  // 使用完整标定视锥，包含偏离画面中心的 cx/cy 和独立 fx/fy，与左侧原图逐像素对应。
  const c = sceneData.camera;
  const near = camera.near;
  camera.projectionMatrix.makePerspective(
    -c.cx * near / c.fx, (c.width - c.cx) * near / c.fx,
    c.cy * near / c.fy, -(c.height - c.cy) * near / c.fy,
    near, camera.far
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

function alignWithImage() {
  // Restore the saved sensor baseline, clearing orbit/pan/zoom state regardless of selected object.
  // 恢复已保存的传感器基准，清除旋转/平移/缩放状态，不受选中哪份实例影响。
  cancelCanvasGesture();
  placeViewPivot(true);
  camera.up.set(0, 1, 0);
  controls.reset();
  camera.position.set(0, 0, 0);
  camera.quaternion.identity();
  camera.zoom = 1;
  // OrbitControls.reset rebuilds a centered projection; reapply the actual calibration afterward.
  // OrbitControls.reset 会重建居中投影，因此最后重新应用真实内参，不能只调用其 reset。
  updateSensorProjection();
  camera.updateMatrixWorld(true);
  render();
}

async function start() {
  try {
    const [metadata, binary] = await Promise.all([fetch('./scene.json'), fetch('./cloud.bin')]);
    if (!metadata.ok || !binary.ok) throw new Error('Saved scene data could not load.');
    sceneData = await metadata.json();
    document.getElementById('scene-label').textContent = sceneData.frame + ' · WBPS ≥ ' + sceneData.threshold;
    for (const label of document.querySelectorAll('[data-count="all"]')) label.textContent = sceneData.instances.length;
    for (const label of document.querySelectorAll('[data-count="accepted"]')) label.textContent = sceneData.instances.filter(row => row.accepted).length;
    const packed = new Float32Array(await binary.arrayBuffer());
    if (packed.length !== sceneData.pointCount * 6) throw new Error('Invalid cloud buffer.');
    image = new Image();
    image.src = './image.png';
    await image.decode();
    renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x091729);
    scene = new THREE.Scene();
    viewPivot = new THREE.Group();
    displayRoot = new THREE.Group();
    displayRoot.matrixAutoUpdate = false;
    viewPivot.add(displayRoot); scene.add(viewPivot);
    const c = sceneData.camera;
    camera = new THREE.PerspectiveCamera(2 * Math.atan(c.height / (2 * c.fy)) * 180 / Math.PI, c.width / c.height, 0.001, 20);
    scene.add(new THREE.HemisphereLight(0xe1f2ff, 0x304059, 2.3));
    const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(0.5, 0.7, 0.2); scene.add(light);
    // Positions are camera-meter XYZ. Flip Y and Z, preserving physical scale and handedness.
    // 输入为相机坐标系米制 XYZ；翻转 Y、Z，保持物理比例与右手系。
    const positions = new Float32Array(sceneData.pointCount * 3);
    const colors = new Float32Array(sceneData.pointCount * 3);
    for (let i = 0; i < sceneData.pointCount; i++) {
      positions.set([packed[i * 6], -packed[i * 6 + 1], -packed[i * 6 + 2]], i * 3);
      colors.set(packed.subarray(i * 6 + 3, i * 6 + 6), i * 3);
    }
    const cloudGeometry = new THREE.BufferGeometry();
    cloudGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    cloudGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    // Bright round samples keep a readable CSS-pixel size instead of shrinking below one pixel.
    // 明亮圆点使用固定 CSS 像素尺寸，避免深度点缩到一个像素以下；保存的 RGB 属性仍保留。
    const pointSprite = document.createElement('canvas');
    pointSprite.width = 32; pointSprite.height = 32;
    const spriteContext = pointSprite.getContext('2d');
    const dot = spriteContext.createRadialGradient(16, 16, 0, 16, 16, 16);
    dot.addColorStop(0, '#ffffff');
    dot.addColorStop(0.65, '#ffffff');
    dot.addColorStop(1, '#ffffff00');
    spriteContext.fillStyle = dot; spriteContext.fillRect(0, 0, 32, 32);
    const pointTexture = new THREE.CanvasTexture(pointSprite);
    // Mipmaps would average a tiny sprite into a faint square; sample its round profile directly.
    // mipmap 会把小尺寸圆点平均成暗方块；直接线性采样圆形轮廓，保留明亮中心。
    pointTexture.generateMipmaps = false;
    pointTexture.minFilter = THREE.LinearFilter;
    cloud = new THREE.Points(cloudGeometry, new THREE.PointsMaterial({size: 1.8, color: 0xfff2df,
      map: pointTexture, alphaTest: 0.1, vertexColors: false, transparent: true,
      opacity: 0.95, depthWrite: false, sizeAttenuation: false, toneMapped: false}));
    // Overlay measured dots after translucent CAD so surfaces do not wash out the sensor samples.
    // 在半透明 CAD 后绘制测量散点，避免面片压暗颗粒；不移动点或改变位姿，仍保留深度测试。
    cloud.renderOrder = 1;
    displayRoot.add(cloud);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(sceneData.vertices, 3));
    geometry.setIndex(sceneData.triangles); geometry.computeVertexNormals();
    const outlineGeometry = new THREE.EdgesGeometry(geometry, 30);
    for (const row of sceneData.instances) {
      const material = new THREE.MeshStandardMaterial({color: row.accepted ? passColor : (row.status === 'suppressed' ? duplicateColor : failColor), transparent: true, opacity: 0.62, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide, depthWrite: false});
      const mesh = new THREE.Mesh(geometry, material);
      const p = row.pose;
      // Row-major object-to-camera pose, meters. No recentering or guessed mesh orientation.
      // 行主序物体到相机位姿，单位米；不重新居中或猜测网格朝向。
      mesh.matrix.set(p[0], p[1], p[2], p[3], -p[4], -p[5], -p[6], -p[7], -p[8], -p[9], -p[10], -p[11], 0, 0, 0, 1);
      const outline = new THREE.LineSegments(outlineGeometry, new THREE.LineBasicMaterial({color: 0xffdc91, transparent: true, opacity: 0.95, depthWrite: false}));
      outline.visible = false;
      mesh.add(outline);
      mesh.matrixAutoUpdate = false; mesh.userData.row = row; mesh.userData.outline = outline;
      displayRoot.add(mesh); meshes.push(mesh);
      const button = document.createElement('button');
      button.type = 'button'; button.dataset.id = row.id; button.dataset.pass = String(row.accepted); button.setAttribute('aria-pressed', 'false');
      button.dataset.status = row.status;
      const label = document.createElement('span'); label.className = 'instance-label'; label.textContent = (language() === 'zh' ? '框 ' : 'Box ') + row.id;
      const score = document.createElement('span'); score.className = 'score'; score.textContent = row.score.toFixed(3);
      button.append(label, score); button.title = 'WBPS ' + row.score.toFixed(6);
      button.addEventListener('click', () => selectInstance(row.id)); instanceButtons.append(button);
    }
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false; controls.minDistance = 0.03; controls.maxDistance = 4;
    // The pivot lies on the sensor optical axis at the first pose's depth; it changes orbit only.
    // 旋转中心位于传感器光轴上，深度取首个位姿；它只控制后续旋转，不倾斜对齐相机。
    controls.target.set(0, 0, -sceneData.instances[0].pose[11]);
    camera.position.set(0, 0, 0);
    controls.update();
    controls.saveState();
    controls.listenToKeyEvents(canvas); controls.addEventListener('change', render);
    photo.addEventListener('click', event => {
      const rect = photo.getBoundingClientRect();
      const x = (event.clientX - rect.left) * photo.width / rect.width;
      const y = (event.clientY - rect.top) * photo.height / rect.height;
      const hit = pickProjectedInstance(x, y);
      if (hit) selectInstance(hit.id);
    });
    for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => {
      mode = button.dataset.mode;
      for (const item of document.querySelectorAll('[data-mode]')) item.setAttribute('aria-pressed', String(item === button));
      const firstAccepted = sceneData.instances.find(row => row.accepted);
      if (mode === 'accepted' && firstAccepted && !sceneData.instances.find(row => row.id === selectedId).accepted) selectInstance(firstAccepted.id);
      else updateScene();
    });
    document.getElementById('cloud-toggle').addEventListener('change', event => { cloud.visible = event.target.checked; render(); });
    meshToggle.addEventListener('change', updateScene);
    wireframeToggle.addEventListener('change', event => { for (const mesh of meshes) mesh.material.wireframe = event.target.checked; render(); });
    for (const button of document.querySelectorAll('[data-align-image]')) button.addEventListener('click', alignWithImage);
    document.getElementById('fullscreen').addEventListener('click', async () => {
      try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
      catch (error) { selection.textContent = language() === 'zh' ? '浏览器未允许全屏。仍可在当前视图中拖动查看。' : 'Fullscreen is unavailable. You can still drag to rotate the view.'; }
    });
    new ResizeObserver(() => {
      const c = sceneData.camera;
      const stage = canvas.parentElement.getBoundingClientRect();
      const width = compact ? Math.min(stage.width, stage.height * c.width / c.height) : canvas.clientWidth;
      if (width > 0) {
        const height = width * c.height / c.width;
        if (compact) {
          // Fit the calibrated image ratio inside the fixed homepage tile without stretching.
          // 在首页固定画框内按标定画幅等比放置，不拉伸点云或 CAD。
          canvas.style.width = width + 'px';
          canvas.style.height = height + 'px';
        }
        renderer.setSize(width, height, false);
        camera.aspect = c.width / c.height;
        updateSensorProjection(); render();
      }
      resizeFrame();
    }).observe(compact ? canvas.parentElement : canvas);
    selectedId = (compact ? sceneData.instances.find(row => row.accepted) : sceneData.instances[0]).id;
    alignWithImage(); createNavigationControls(); updateScene(); loading.hidden = true;
    root.dataset.ready = 'true'; resizeFrame();
  } catch (error) {
    root.dataset.ready = 'error';
    loading.hidden = false;
    updateSelection(); drawPhoto();
    console.error(error); resizeFrame();
  }
}

syncPreferences();
try { if (window.parent !== window) new MutationObserver(syncPreferences).observe(window.parent.document.documentElement, {attributes: true, attributeFilter: ['data-lang', 'data-theme']}); } catch (error) { /* Standalone viewer. / 独立查看。 */ }
new ResizeObserver(resizeFrame).observe(document.body);
start();
