"""Validate the deployed path layout, copied assets and README code coverage."""
from html.parser import HTMLParser
from pathlib import Path
import json
import re
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
BASE = 'https://wangyulin-seu.github.io/HCCEPose/'


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.refs, self.ids, self.pre_count, self.lang, self.h1_count = [], set(), 0, '', 0
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'html':
            self.lang = attrs.get('lang', '')
        if tag == 'pre':
            self.pre_count += 1
        if tag == 'h1':
            self.h1_count += 1
        if tag in ('a', 'link') and attrs.get('href'):
            self.refs.append(attrs['href'])
        if tag in ('img', 'script') and attrs.get('src'):
            self.refs.append(attrs['src'])


def code_blocks(text):
    # All current README code fences use backticks; compare actual block bodies.
    return re.findall(r'^```[^\n]*\n(.*?)^```\s*$', text, re.M | re.S)


def main():
    pages = {p.resolve(): Page(p) for p in SITE.rglob('*.html')}
    errors = []
    for path, page in pages.items():
        rel = path.relative_to(SITE.resolve()).as_posix()
        current = BASE + (rel[:-10] if rel.endswith('index.html') else rel)
        for ref in page.refs:
            url = urlsplit(urljoin(current, ref))
            if url.netloc != urlsplit(BASE).netloc or url.scheme not in ('http', 'https'):
                continue
            if not url.path.startswith('/HCCEPose/'):
                errors.append(f'{rel}: outside project base path: {ref}')
                continue
            target = SITE / unquote(url.path.removeprefix('/HCCEPose/'))
            if target.is_dir() or url.path.endswith('/'):
                target = target / 'index.html'
            target = target.resolve()
            if not target.is_file():
                errors.append(f'{rel}: missing link or asset: {ref}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{rel}: missing anchor: {ref}')

    for lang, source in [('en', 'README.md'), ('zh', 'README_CN.md')]:
        generated = sorted((ROOT / 'docs/content' / lang).glob('*.md'))
        original = code_blocks((ROOT / source).read_text(encoding='utf-8'))
        migrated = [block for p in generated for block in code_blocks(p.read_text(encoding='utf-8'))]
        if sorted(original) != sorted(migrated):
            errors.append(f'{source}: fenced code changed or missing')
        count = 0
        for p in generated:
            output = SITE / ('' if lang == 'en' else lang)
            output = output / ('index.html' if p.stem == 'index' else p.stem + '/index.html')
            rendered = pages.get(output.resolve())
            if rendered is None:
                errors.append(f'Missing page: {output}')
                continue
            if rendered.lang != lang or rendered.h1_count != 1:
                errors.append(f'{output}: incorrect language or main heading')
            count += rendered.pre_count
        if count != len(original):
            errors.append(f'{source}: {len(original)} source code blocks but {count} rendered')
        print(f'{lang}: {len(generated)} pages; {len(original)} unchanged code blocks; {count} rendered blocks')

    search = json.loads((SITE / 'search/search_index.json').read_text(encoding='utf-8'))
    locations = [entry['location'] for entry in search['docs']]
    if not any(location.startswith('zh/') for location in locations):
        errors.append('Chinese pages missing from combined search index')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Passed: {len(pages)} HTML pages, local links, anchors, assets, bilingual search index and README code preservation.')


if __name__ == '__main__':
    main()
