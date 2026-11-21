"""Check searchable article text and accessible source references.
检查可搜索正文与可访问的来源引用。
"""
import importlib.util
from pathlib import Path

import pytest
from bs4 import BeautifulSoup


project_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('docs_build', project_dir / 'docs' / 'build.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize('lang', ['zh', 'en'])
def test_search_keeps_article_without_playback_controls(lang):
    _, pages, slugs = builder.read_navigation()
    page = next(page for page in pages if page['slug'] == 'video')
    content, _, entry = builder.render_source(page, lang, slugs, {})
    soup = BeautifulSoup(content, 'html.parser')
    assert soup.select_one('video[aria-busy="true"]') is not None
    assert soup.select_one('.video-loading-spinner') is not None
    assert soup.select_one('.video-loading-retry') is not None
    assert 'C1–C6' in entry['text']
    # Both translations remain available to the player but absent from the index.
    # 播放器保留两种提示文案，索引只包含所选语言正文。
    for label in ('视频加载中', 'Loading video', '重试', 'Retry'):
        assert label in soup.get_text()
        assert label not in entry['text']
    if lang == 'en':
        assert '逐帧' not in entry['text']
    else:
        assert 'Poses across frames' not in entry['text']


@pytest.mark.parametrize('lang', ['zh', 'en'])
def test_citation_links_resolve_to_license_and_source_files(lang):
    _, pages, slugs = builder.read_navigation()
    page = next(page for page in pages if page['slug'] == 'citation')
    assets = {}
    content, _, _ = builder.render_source(page, lang, slugs, assets)
    soup = BeautifulSoup(content, 'html.parser')
    for filename in ('LICENSE', 'PUBLIC_SOURCES.md'):
        destination = 'assets/' + filename
        assert soup.find('a', href='../' + destination) is not None
        assert Path(assets[destination]) == project_dir / filename
        assert Path(assets[destination]).is_file()
