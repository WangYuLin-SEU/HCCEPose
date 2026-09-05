"""Build bilingual documentation from README.md and README_CN.md.

The READMEs remain the source of truth. No training packages are imported.
Section boundaries are exact headings; changed/missing headings fail the build.
"""
from html import escape, unescape
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit

from mkdocs.plugins import event_priority

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'docs' / 'content'
REPO = 'https://github.com/WangYuLin-SEU/HCCEPose'
HF = 'https://huggingface.co/datasets/SEU-WYL/HccePose'

# slug, English boundary, Chinese boundary, English title, Chinese title
SECTIONS = [
    ('index', '## 🧩 Introduction', '## 🧩 简介', 'HccePose (BF)', 'HccePose (BF)'),
    ('updates', '## ✨ Update', '## ✨ 更新', 'Updates', '更新记录'),
    ('installation', '## 🔧 Environment Setup', '## 🔧 环境配置', 'Installation', '环境配置'),
    ('downloads', '### 📥 Bulk download from Hugging Face (optional)', '### 📥 从 Hugging Face 批量下载（可选）', 'Downloads', '资源下载'),
    ('minimal-inference', '## 🎯 Minimal setup (inference demo)', '## 🎯 最小推理环境（Bin-Picking RGB）', 'Minimal inference', '最小推理环境'),
    ('troubleshooting', '### 🩹 Troubleshooting (common)', '### 🩹 故障排查（常见）', 'Troubleshooting', '故障排查'),
    ('objects', '## 🧱 Custom Dataset and Training', '## 🧱 自定义数据集及训练', 'Object preparation', '物体预处理与对称分析'),
    ('rendering', '#### 🔥 Rendering the PBR Dataset', '#### 🔥 渲染 PBR 数据集', 'PBR rendering', 'PBR 数据渲染'),
    ('detector', '#### 🚀 Training the 2D Detector', '#### 🚀 训练 2D 检测器', '2D detector', '2D 检测器训练'),
    ('labels', '#### 🧩 Preparation of Front–Back Surface Labels', '#### 🧩 物体正背面标签制备', 'Surface labels', '正背面标签制备'),
    ('training', '#### 🚀 Training HccePose(BF)', '#### 🚀 训练 HccePose(BF)', 'HccePose training', 'HccePose 训练'),
    ('quick-start', '## ✏️ Quick Start', '## ✏️ 快速开始', 'RGB demo', 'RGB 图像示例'),
    ('video', '#### 🎥 6D Pose Estimation in Videos', '#### 🎥 视频的6D位姿估计', 'Video', '视频示例'),
    ('rgbd', '#### 📷 RGB-D refinement (FoundationPose / MegaPose)', '#### 📷 RGB-D 微调（FoundationPose / MegaPose）', 'RGB-D inputs', 'RGB-D 输入'),
    ('foundationpose', '#### 📸 Example: FoundationPose RGB-D refinement', '#### 📸 示例：FoundationPose RGB-D 微调', 'FoundationPose refinement', 'FoundationPose 微调'),
    ('megapose', '#### 📸 Example: MegaPose refinement (RGB-D branch)', '#### 📸 示例：MegaPose 微调（RGB-D 分支）', 'MegaPose refinement', 'MegaPose 微调'),
    ('comparison', '#### 📸 Example: HccePose vs FoundationPose vs MegaPose (comparison & depth fusion)', '#### 📸 示例：HccePose / FoundationPose / MegaPose 对比与深度融合', 'Refinement comparison', '微调对比与深度融合'),
    ('bop', '## 🧪 BOP Challenge Testing', '## 🧪 BOP挑战测试', 'BOP evaluation', 'BOP 评测'),
    ('roadmap', '## 📅 Update Plan', '## 📅 更新计划', 'Historical roadmap', '历史更新计划'),
    ('results', '## 🏆 BOP LeaderBoards', '## 🏆 BOP榜单', 'Leaderboards', '历史榜单'),
    ('citation', '## Acknowledgments', '## 致谢', 'Acknowledgments and citation', '致谢与引用'),
]
ANCHORS = {'environment-setup': 'installation', 'minimal-inference-setup': 'minimal-inference'}


def prose_only(text, transform):
    """Keep fenced code byte-for-byte, including code comments and links."""
    result, prose, fence = [], [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            result.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
        elif marker:
            result.append(transform(''.join(prose)))
            prose = []
            result.append(line)
            fence = marker[1]
        else:
            prose.append(line)
    result.append(transform(''.join(prose)))
    if fence:
        raise ValueError('Unclosed fenced code block in README')
    return ''.join(result)


def transform_prose(text):
    # Whole chapters are readable without expanding GitHub-specific wrappers.
    text = re.sub(r'</?details\b[^>]*>', '', text)
    text = re.sub(r'<summary>(.*?)</summary>', lambda m: '\n## ' + m[1] + '\n' if m[1].startswith(('Optional:', '可选：')) else '', text)
    text = re.sub(r'<a id="(?:environment-setup|minimal-inference-setup)"></a>', '', text)
    text = re.sub(r'^(#{2,6})\s+([^\n]+)', lambda m: '## ' + re.sub(r'^[^\w\u4e00-\u9fff]+', '', m[2]).strip(), text, flags=re.M)

    def link_target(target):
        if urlsplit(target).scheme or target.startswith('//'):
            return target
        if target.startswith('#'):
            anchor = target[1:]
            if anchor not in ANCHORS:
                raise ValueError(f'Unmapped README anchor: {target}')
            return ANCHORS[anchor] + '.md' + target
        path = target.removeprefix('./').lstrip('/')
        if path.startswith('hf-dataset-card/'):
            return HF
        if not (ROOT / path.split('#')[0]).exists():
            raise ValueError(f'Missing repository link target: {target}')
        return REPO + '/blob/main/' + path

    text = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + link_target(m[2]) + m[3], text)

    def image_tag(match):
        tag = match[0]
        src = re.search(r'src=["\']([^"\']+)["\']', tag)[1]
        if urlsplit(src).scheme:
            return tag
        path = ROOT / src
        if not path.is_file():
            raise ValueError(f'Missing README image: {src}')
        dest = CONTENT / 'assets' / src
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        alt = re.search(r'alt=["\']([^"\']*)["\']', tag)
        alt = unescape(alt[1]) if alt else path.stem.replace('_', ' ')
        width = re.search(r'width=["\']?([\d%]+)', tag)
        width = f' width="{width[1]}"' if width else ''
        return f'\n![{escape(alt)}](../assets/{src}){{ loading="lazy"{width} }}\n'

    text = re.sub(r'<img\b[^>]*>', image_tag, text)
    # Let Python-Markdown process images in the original centered HTML blocks.
    text = re.sub(r'<(div|p)(\s+[^>]*)?>', lambda m: '<' + m[1] + (m[2] or '') + ' markdown="1">', text)
    return text


def generate():
    # These directories contain generated files only. Clear them so renamed
    # chapters and removed figures cannot linger in local rebuilds.
    for name in ('en', 'zh', 'assets'):
        generated = CONTENT / name
        if generated.exists():
            shutil.rmtree(generated)
    for lang, source, boundary_idx, title_idx in [('en', 'README.md', 1, 3), ('zh', 'README_CN.md', 2, 4)]:
        text = (ROOT / source).read_text(encoding='utf-8')
        text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
        positions = []
        for section in SECTIONS:
            matches = list(re.finditer('^' + re.escape(section[boundary_idx]) + r'\s*$', text, re.M))
            if len(matches) != 1:
                raise ValueError(f'{source}: expected exactly one heading {section[boundary_idx]!r}')
            positions.append(matches[0].start())
        if positions != sorted(positions):
            raise ValueError(f'{source}: README section order changed; update SECTIONS')
        folder = CONTENT / lang
        folder.mkdir(parents=True, exist_ok=True)
        for index, section in enumerate(SECTIONS):
            slug, title = section[0], section[title_idx]
            end = positions[index + 1] if index + 1 < len(positions) else len(text)
            raw = text[positions[index]:end].split('\n', 1)[1]
            body = prose_only(raw, transform_prose).strip()
            body = re.sub(r'(?:\s*---\s*)+$', '', body).strip()
            prefix = ''
            for anchor, anchor_slug in ANCHORS.items():
                if slug == anchor_slug:
                    prefix += f'<a id="{anchor}"></a>\n\n'
            if slug in ('roadmap', 'results'):
                note = ('以下内容保留自 README，属于历史记录；其中的计划日期和榜单截图不代表当前状态。' if lang == 'zh' else 'Preserved from the README as a historical record. Roadmap dates and leaderboard screenshots do not represent current status.')
                prefix += f'> {note}\n\n'
            if slug == 'index':
                links = ('[最小推理环境](minimal-inference.md) · [RGB 示例](quick-start.md) · [自定义物体训练](objects.md)' if lang == 'zh' else '[Minimal inference](minimal-inference.md) · [RGB demo](quick-start.md) · [Train custom objects](objects.md)')
                prefix += f'{links}\n\n[Paper](https://arxiv.org/abs/2510.10177) · [Datasets & weights]({HF}) · [GitHub]({REPO})\n\n'
            label = '源文档' if lang == 'zh' else 'Source'
            page = f'# {title}\n\n{prefix}{body}\n\n---\n\n{label}: [{source}]({REPO}/blob/main/{source})\n'
            (folder / f'{slug}.md').write_text(page, encoding='utf-8')
    styles = CONTENT / 'assets' / 'stylesheets'
    styles.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'docs/stylesheets/extra.css', styles / 'extra.css')


@event_priority(100)
def on_pre_build(config):
    generate()


if __name__ == '__main__':
    generate()
    print(f'Generated {len(SECTIONS) * 2} documentation pages from the two READMEs.')
