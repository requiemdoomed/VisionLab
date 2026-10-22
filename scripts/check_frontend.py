"""Check the inline frontend without a build step."""
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if 'id' in attributes:
            self.ids.append(attributes['id'])
        if tag == 'script' and 'src' in attributes:
            raise ValueError('The interface must not need an external script.')


def main():
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    parser = Elements(); parser.feed(html)
    assert len(parser.ids) == len(set(parser.ids)), 'Duplicate HTML IDs'
    scripts = re.findall(r'<script>([\s\S]*?)</script>', html)
    assert len(scripts) == 1
    missing = set(re.findall(r"\$\('([^']+)'\)", scripts[0])) - set(parser.ids)
    assert not missing, f'Missing controls: {missing}'
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'app.js'
        path.write_text(scripts[0], encoding='utf-8')
        subprocess.run(['node', '--check', str(path)], check=True)
    print(f'Frontend passed: {len(parser.ids)} unique IDs; {len(html.encode())} bytes; inline CSS and JS.')


if __name__ == '__main__':
    main()
