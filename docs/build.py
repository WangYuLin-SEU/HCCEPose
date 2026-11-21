"""Build and check the local static documentation. / 构建并检查本地静态文档。

Run from any directory: python /path/to/HCCEPose/docs/build.py
可从任意目录运行；源码路径始终相对于本脚本解析。
"""

import html
import json
import os
import re
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import markdown
from bs4 import BeautifulSoup
from markdown.extensions.toc import slugify_unicode
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


# Source and output roots follow this script, independent of the working directory.
# 源码与产物根目录从当前脚本推导，不依赖启动时的工作目录。
docs_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(docs_dir)
pages_dir = os.path.join(project_dir, "pages")
output_dir = os.path.join(pages_dir, "docs")
theme_dir = os.path.join(docs_dir, "theme")
languages = ("zh", "en")


def bilingual(labels):
    """Keep translated navigation labels in one stable slot. / 双语导航文案共用稳定占位。"""
    pieces = ['<span class="language-pair">']
    for lang in languages:
        language_tag = "zh-CN" if lang == "zh" else "en"
        pieces.append(
            '<span class="' + lang + '" lang="' + language_tag + '">'
            + html.escape(labels[lang]) + "</span>"
        )
    pieces.append("</span>")
    return "".join(pieces)


def add_video_loading_feedback(soup):
    """Paint feedback before deferred playback scripts run. / 播放脚本执行前即可绘制加载反馈。"""
    for video in soup.find_all("video"):
        if video.find_parent(class_="video-player") is not None:
            continue
        wrapper = soup.new_tag("div", attrs={"class": "video-player"})
        if video.has_attr("hidden"):
            wrapper["hidden"] = ""
        video.wrap(wrapper)
        video["aria-busy"] = "true"
        # Pre-render both translations in the same slot; JavaScript reuses these nodes.
        # 预渲染双语共同占位；JavaScript 直接复用节点，不等待资源请求后再创建提示。
        overlay = BeautifulSoup(
            '<div class="video-loading" data-state="loading" role="status" aria-live="polite" aria-atomic="true">'
            '<div class="video-loading-card"><span class="video-loading-spinner" aria-hidden="true"></span>'
            '<div class="video-loading-copy"><strong class="video-loading-title">'
            + bilingual({"zh": "视频加载中", "en": "Loading video"})
            + '</strong><span class="video-loading-detail">'
            + bilingual({"zh": "即将自动播放", "en": "Starting shortly"})
            + '</span><button class="video-loading-retry" type="button" hidden>'
            + bilingual({"zh": "重试", "en": "Retry"})
            + '</button></div></div></div>',
            "html.parser",
        )
        wrapper.append(overlay.div)


def read_navigation():
    """Validate the explicit reading order. / 检查显式声明的阅读顺序。"""
    path = os.path.join(docs_dir, "navigation.json")
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    pages = []
    slugs = set()
    for group in groups:
        for lang in languages:
            if not group["label"].get(lang):
                raise ValueError("Missing navigation group label: " + lang)
        for page in group["pages"]:
            slug = page["slug"]
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
                raise ValueError("Invalid page slug: " + repr(slug))
            if slug in slugs:
                raise ValueError("Duplicate page slug: " + slug)
            for lang in languages:
                if not page["title"].get(lang):
                    raise ValueError("Missing page title: " + slug + " / " + lang)
                source_path = os.path.join(docs_dir, lang, slug + ".md")
                if not os.path.isfile(source_path):
                    raise FileNotFoundError("Missing documentation source: " + source_path)
            pages.append({**page, "group": group["label"]})
            slugs.add(slug)
    if not pages or pages[0]["slug"] != "index":
        raise ValueError("The first documentation page must have slug 'index'.")
    return groups, pages, slugs


def render_source(page, lang, slugs, assets):
    """Render one language and resolve local links. / 渲染一种语言并解析本地链接。"""
    source_path = os.path.join(docs_dir, lang, page["slug"] + ".md")
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    rendered = markdown.markdown(
        source,
        extensions=["fenced_code", "tables", "toc", "attr_list", "md_in_html"],
        extension_configs={"toc": {"permalink": False, "slugify": slugify_unicode}},
    )
    soup = BeautifulSoup(rendered, "html.parser")
    if len(soup.find_all("h1")) != 1:
        raise ValueError("Each source needs exactly one H1: " + source_path)

    # Highlight at build time; copying must preserve the original code verbatim.
    # 构建时生成语法颜色；复制内容必须与原始代码完全一致，无需浏览器加载高亮库。
    for code in soup.select("pre > code"):
        code_text = code.get_text()
        code_language = next((name[9:] for name in code.get("class", []) if name.startswith("language-")), "text")
        try:
            lexer = get_lexer_by_name(code_language, stripnl=False, ensurenl=False)
        except ClassNotFound:
            lexer = get_lexer_by_name("text", stripnl=False, ensurenl=False)
        colored_html = highlight(code_text, lexer, HtmlFormatter(nowrap=True))
        if not code_text.endswith("\n"):
            colored_html = colored_html.removesuffix("\n")
        # Parse inside pre so whitespace-only tokens keep every blank line.
        # 在 pre 内解析，保留纯空白 token 中的连续空行。
        colored_code = BeautifulSoup("<pre>" + colored_html + "</pre>", "html.parser").pre
        if colored_code.get_text() != code_text:
            raise ValueError("Highlighting changed code text: " + source_path)
        code.clear()
        code["class"] = code.get("class", []) + ["syntax-highlight"]
        for token in list(colored_code.contents):
            code.append(token)

        # Keep language and copy controls outside the horizontally scrolling code.
        # 语言标记和复制按钮置于代码横向滚动区外，长行不会遮住操作入口。
        block = soup.new_tag("div", attrs={"class": "code-block", "data-code-language": code_language})
        pre = code.parent
        pre.wrap(block)
        pre["tabindex"] = "0"
        pre["aria-label"] = code_language + (" 代码" if lang == "zh" else " code")
        toolbar = soup.new_tag("div", attrs={"class": "code-toolbar"})
        language_label = soup.new_tag("span", attrs={"class": "code-label"})
        language_label.string = code_language.upper()
        toolbar.append(language_label)
        block.insert(0, toolbar)

    # Language prefixes avoid duplicate IDs in the bilingual document.
    # 双语正文使用不同 ID 前缀，避免同页锚点重名。
    for node in soup.find_all(id=True):
        node["id"] = lang + "-" + node["id"]
    for node in soup.find_all(["a", "img", "video", "source", "iframe"]):
        for attribute in ("href", "src", "poster"):
            value = node.get(attribute)
            if not value:
                continue
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue
            if not url.path:
                if url.fragment:
                    node[attribute] = "#" + lang + "-" + unquote(url.fragment)
                continue
            local_path = os.path.abspath(os.path.join(os.path.dirname(source_path), unquote(url.path)))
            if os.path.commonpath([project_dir, local_path]) != project_dir:
                raise ValueError("Link escapes the project: " + source_path + " -> " + value)
            # Root Markdown references are downloadable files; localized docs become pages.
            # 根目录 Markdown 引用作为文件提供；分语言文档引用转换为网页。
            if url.path.endswith(".md") and os.path.dirname(local_path) != project_dir:
                target_slug = os.path.splitext(os.path.basename(local_path))[0]
                if target_slug not in slugs:
                    raise ValueError("Unknown documentation page: " + source_path + " -> " + value)
                target_lang = os.path.basename(os.path.dirname(local_path))
                if target_lang not in languages:
                    raise ValueError("Link Markdown sources through docs/en or docs/zh: " + value)
                fragment = target_lang + "-" + unquote(url.fragment) if url.fragment else ""
                node[attribute] = urlunsplit(("", "", target_slug + ".html", "lang=" + target_lang, fragment))
            elif node.name != "iframe" and url.path.endswith(".html") and os.path.basename(url.path)[:-5] in slugs:
                fragment = lang + "-" + unquote(url.fragment) if url.fragment else ""
                node[attribute] = urlunsplit(("", "", os.path.basename(url.path), "lang=" + lang, fragment))
            else:
                if not os.path.isfile(local_path):
                    raise FileNotFoundError("Missing referenced resource: " + source_path + " -> " + value)
                relative_path = os.path.relpath(local_path, project_dir).replace(os.sep, "/")
                destination = "assets/" + relative_path
                assets[destination] = local_path
                node[attribute] = urlunsplit(("", "", "../" + quote(destination, safe="/"), url.query, url.fragment))

    # Search indexes contain article text, before bilingual playback controls.
    # 搜索索引先提取正文，避免双语播放器提示混入任一语言的搜索结果。
    search_text = soup.get_text(" ", strip=True)
    add_video_loading_feedback(soup)
    for image in soup.find_all("img"):
        image["loading"] = "lazy"
        image["decoding"] = "async"
        if not image.get("alt"):
            raise ValueError("Images need descriptive alt text: " + source_path + " -> " + image.get("src", ""))
        if image.find_parent("a") is None:
            original = soup.new_tag("a", href=image["src"], target="_blank", rel="noopener")
            original["class"] = "image-original"
            original["aria-label"] = ("查看原图：" if lang == "zh" else "Open original image: ") + image["alt"]
            image.wrap(original)
    for table in soup.find_all("table"):
        scroll = soup.new_tag("div", attrs={"class": "table-scroll", "tabindex": "0"})
        scroll["role"] = "region"
        scroll["aria-label"] = "横向滚动表格" if lang == "zh" else "Horizontally scrollable table"
        table.wrap(scroll)
    for node in soup.find_all("a", href=True):
        if node["href"].startswith(("https://", "http://")):
            node["rel"] = "noopener noreferrer"
    outline = []
    for heading in soup.find_all(["h2", "h3"]):
        outline.append(
            '<li class="outline-' + heading.name + '"><a href="#'
            + html.escape(heading["id"], quote=True) + '">'
            + html.escape(heading.get_text(" ", strip=True)) + "</a></li>"
        )
    content = '<div class="doc-language ' + lang + '">' + str(soup) + "</div>"
    contents = '<ul class="' + lang + '">' + "".join(outline) + "</ul>"
    index = {
        "slug": page["slug"],
        "lang": lang,
        "title": page["title"][lang],
        "text": search_text,
        "url": page["slug"] + ".html?lang=" + lang,
    }
    return content, contents, index


def render_navigation(groups, current_slug):
    """Build ordinary links with collapsible groups. / 生成带可折叠分组的普通链接。"""
    result = []
    for group in groups:
        is_current = any(page["slug"] == current_slug for page in group["pages"])
        result.append('<details class="nav-group"' + (" open" if is_current else "") + ">")
        result.append("<summary>" + bilingual(group["label"]) + "</summary><ul>")
        for page in group["pages"]:
            active = ' aria-current="page"' if page["slug"] == current_slug else ""
            result.append('<li><a href="' + page["slug"] + '.html"' + active + ">" + bilingual(page["title"]) + "</a></li>")
        result.append("</ul></details>")
    return "".join(result)


def render_pagination(pages, index, position):
    """Use the same linear order as the sidebar. / 上下页与侧栏共用线性顺序。"""
    pieces = ['<nav class="pagination pagination-' + position + '" aria-label="上下页 / Previous and next pages">']
    for offset, direction, symbol, labels in (
        (-1, "prev", "←", {"zh": "上一页", "en": "Previous"}),
        (1, "next", "→", {"zh": "下一页", "en": "Next"}),
    ):
        target_index = index + offset
        if target_index < 0 or target_index >= len(pages):
            pieces.append('<span class="pagination-empty"></span>')
            continue
        target = pages[target_index]
        pieces.append(
            '<a rel="' + direction + '" href="' + target["slug"] + '.html">'
            + '<small><span aria-hidden="true">' + symbol + "</span> " + bilingual(labels) + "</small>"
            + "<strong>" + bilingual(target["title"]) + "</strong></a>"
        )
    pieces.append("</nav>")
    return "".join(pieces)


def check_output(documents, assets):
    """Check links against generated pages and copied files. / 检查生成页面及复制资源的内部引用。"""
    ids_by_path = {}
    parsed = {}
    for relative_path, document in documents.items():
        soup = BeautifulSoup(document, "html.parser")
        all_ids = [node["id"] for node in soup.find_all(id=True)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate HTML IDs in " + relative_path)
        ids_by_path[relative_path] = set(all_ids)
        parsed[relative_path] = soup
    available = set(documents) | set(assets)
    errors = []
    for relative_path, soup in parsed.items():
        for node in soup.find_all(["a", "img", "script", "link", "video", "source", "iframe"]):
            for attribute in ("href", "src", "poster", "data-animation"):
                value = node.get(attribute)
                if not value:
                    continue
                url = urlsplit(value)
                if url.scheme or url.netloc:
                    continue
                destination = os.path.normpath(os.path.join(os.path.dirname(relative_path), unquote(url.path))) if url.path else relative_path
                destination = destination.replace(os.sep, "/")
                if destination not in available:
                    errors.append(relative_path + " -> missing " + value)
                elif url.fragment and destination in ids_by_path and unquote(url.fragment) not in ids_by_path[destination]:
                    errors.append(relative_path + " -> missing anchor " + value)
    if errors:
        raise ValueError("Documentation link validation failed:\n" + "\n".join(errors))


def main():
    """Build static pages only; no upload or deployment. / 只构建静态页面，不上传或部署。"""
    groups, pages, slugs = read_navigation()
    with open(os.path.join(theme_dir, "template.html"), encoding="utf-8") as handle:
        template = handle.read()
    assets = {
        "assets/docs/style.css": os.path.join(theme_dir, "style.css"),
        "assets/docs/site.js": os.path.join(theme_dir, "site.js"),
        "assets/docs/video.js": os.path.join(theme_dir, "video.js"),
        "assets/docs/landing.css": os.path.join(theme_dir, "landing.css"),
        "assets/docs/landing.js": os.path.join(theme_dir, "landing.js"),
        "assets/docs/motion.css": os.path.join(theme_dir, "motion.css"),
        "assets/docs/motion.js": os.path.join(theme_dir, "motion.js"),
        "assets/docs/assets/hccepose-contours.svg": os.path.join(docs_dir, "assets", "hccepose-contours.svg"),
        "assets/media/public-result.mp4": os.path.join(project_dir, "media", "public-result.mp4"),
        "assets/media/public-poster.jpg": os.path.join(project_dir, "media", "public-poster.jpg"),
        "assets/show_vis/fig2.jpg": os.path.join(project_dir, "show_vis", "fig2.jpg"),
        "assets/show_vis/IMG_20251007_165718_show_6d_vis1.jpg": os.path.join(project_dir, "show_vis", "IMG_20251007_165718_show_6d_vis1.jpg"),
    }
    # Bundle the saved RGB-D display without importing or executing the pose estimator.
    # 仅打包已保存的 RGB-D 展示，不导入或执行位姿估计器。
    for filename in ("index.html", "viewer.js", "viewer.css", "three.module.min.js",
                     "OrbitControls.js", "TransformControls.js", "THREE-LICENSE.txt", "THIRD_PARTY.md",
                     "image.png", "cloud.bin", "scene.json"):
        assets["assets/docs/assets/wapr-viewer/" + filename] = os.path.join(docs_dir, "assets", "wapr-viewer", filename)
    documents = {}
    search_entries = []
    for index, page in enumerate(pages):
        contents = []
        outlines = []
        for lang in languages:
            content, outline, search_entry = render_source(page, lang, slugs, assets)
            contents.append(content)
            outlines.append(outline)
            search_entries.append(search_entry)
        replacements = {
            "CANONICAL_URL": "https://wangyulin-seu.github.io/hccepose/docs/" + page["slug"] + ".html",
            "TITLE_ZH": html.escape(page["title"]["zh"], quote=True),
            "TITLE_EN": html.escape(page["title"]["en"], quote=True),
            "DESCRIPTION": html.escape(page["title"]["zh"] + " / " + page["title"]["en"] + " — HCCEPose documentation", quote=True),
            "NAVIGATION": render_navigation(groups, page["slug"]),
            "GROUP_LABEL": bilingual(page["group"]),
            "CONTENT": "\n".join(contents),
            "OUTLINE": "\n".join(outlines),
            "PAGINATION_TOP": render_pagination(pages, index, "top"),
            "PAGINATION_BOTTOM": render_pagination(pages, index, "bottom"),
        }
        document = re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: replacements[match.group(1)], template)
        documents["docs/" + page["slug"] + ".html"] = document
    # The project homepage and documentation keep separate, stable entry addresses.
    # 宣传首页与使用文档分别保留稳定入口，已有文档链接无需重定向。
    with open(os.path.join(theme_dir, "landing.html"), encoding="utf-8") as handle:
        landing = BeautifulSoup(handle.read(), "html.parser")
    add_video_loading_feedback(landing)
    documents["index.html"] = str(landing)
    check_output(documents, assets)
    # Remove only this builder's output after validating every source reference.
    # 全部引用验证完成后，仅清理本构建器的输出，避免残留旧资源。
    if os.path.isdir(pages_dir):
        shutil.rmtree(pages_dir)
    # Validate every source first so missing content cannot leave a partial rebuild.
    # 完成全部源文件与链接检查后再写产物，避免缺页导致半次构建。
    for relative_path, document in documents.items():
        destination = os.path.join(pages_dir, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(document)
    for relative_path, source_path in assets.items():
        destination = os.path.join(pages_dir, relative_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source_path, destination)
    with open(os.path.join(output_dir, "search-index.json"), "w", encoding="utf-8") as handle:
        json.dump(search_entries, handle, ensure_ascii=False, separators=(",", ":"))
    print("Built the project homepage,", len(pages), "bilingual documentation pages and", len(assets), "assets.")
    print("Internal links, anchors, image descriptions and resources: passed.")
    print("Preview: python -m http.server 8766 --bind 127.0.0.1 --directory " + pages_dir)


if __name__ == "__main__":
    main()
