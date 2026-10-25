"""Download a web page or PDF that an uploaded document links to, safely.

The server fetches a link on the operator's click. The address comes from a
document written by someone else, so it is treated as hostile until checked:

* https only, no password in the address, port 443 only;
* the name must resolve, and every address it resolves to must be a public one
  (never this machine, a private network, a link-local or reserved address,
  which is where a cloud provider's secrets and the agent's own services sit);
* a redirect is followed by hand and every hop is checked the same way, so a
  public page cannot send the server on to a private one;
* at most 30 seconds in all (a server that trickles bytes cannot hold the
  request open) and a few megabytes, and only a web page, plain text
  or a PDF;
* what comes back is text only: scripts and styles are dropped, and nothing in
  it is ever run.

One gap remains and is stated plainly: the address is checked just before the
request, and the request looks the name up again, so a hostile name server
that answers differently the second time could slip past. The fetch is
operator-initiated, one link at a time, and what it returns is only ever
treated as untrusted text that is quote-checked before anything is kept.
"""
import ipaddress
import os
import socket
import tempfile
import time
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 3_000_000
MAX_TEXT_CHARS = 200_000
TIMEOUT_SECONDS = 15
DEADLINE_SECONDS = 30            # the whole download, redirects included
MAX_REDIRECTS = 3
MIN_TEXT_CHARS = 200
ACCEPTED = ('text/html', 'application/xhtml+xml', 'text/plain', 'application/pdf')
USER_AGENT = 'Mozilla/5.0 (compatible; trading-agent-research-reader)'
_SKIP = {'script', 'style', 'noscript', 'svg', 'template', 'iframe', 'head'}
_BREAK = {'p', 'br', 'div', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'section', 'article'}


class FetchError(Exception):
    """A reason, in plain words, why a link was not followed."""


def _resolve(host: str) -> List[str]:
    try:
        return sorted({info[4][0] for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    except OSError as e:
        raise FetchError(f'the address {host} could not be found ({e.__class__.__name__})')


def _public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split('%')[0])
    if getattr(addr, 'ipv4_mapped', None):
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def check_url(url: str, resolver: Optional[Callable[[str], List[str]]] = None) -> str:
    """The address, if it is safe to fetch; else FetchError saying why."""
    parts = urlsplit(str(url or '').strip())
    if parts.scheme != 'https':
        raise FetchError('only https links are followed')
    if parts.username or parts.password:
        raise FetchError('a link with a password in it is not followed')
    host = parts.hostname
    if not host:
        raise FetchError('the link has no address')
    try:
        port = parts.port
    except ValueError:
        raise FetchError('the link has an invalid port')
    if port not in (None, 443):
        raise FetchError('only the standard https port is followed')
    ips = (resolver or _resolve)(host)
    if not ips:
        raise FetchError(f'the address {host} could not be found')
    for ip in ips:
        if not _public(ip):
            raise FetchError(f'{host} points to a private or internal address, so it is not followed')
    return parts.geturl()


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.title = ''
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self._in_title = True
        if tag in _SKIP and tag != 'head':
            self._skip += 1
        if tag in _BREAK:
            self.out.append('\n')

    def handle_endtag(self, tag):
        if tag == 'title':
            self._in_title = False
        if tag in _SKIP and tag != 'head' and self._skip:
            self._skip -= 1
        if tag in _BREAK:
            self.out.append('\n')

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> Dict[str, str]:
    """Visible text and the page title; scripts, styles and markup are dropped."""
    p = _Text()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    lines = [' '.join(line.split()) for line in ''.join(p.out).splitlines()]
    return {'title': ' '.join(p.title.split())[:160], 'text': '\n'.join(l for l in lines if l)}


def _pdf_text(data: bytes) -> str:
    from src.utils import pdf_parser
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'linked.pdf')
        with open(path, 'wb') as f:
            f.write(data)
        got = pdf_parser.extract_all(path)
    return str(got.get('text') or '')


def fetch(url: str, session: Any = None, resolver: Optional[Callable[[str], List[str]]] = None,
          max_bytes: int = MAX_BYTES, deadline_s: float = DEADLINE_SECONDS,
          clock: Callable[[], float] = time.monotonic) -> Dict[str, Any]:
    """The readable text at `url`: {'url' (where it ended up), 'title', 'text', 'content_type'}.
    Raises FetchError with a plain reason."""
    if session is None:
        import requests
        session = requests.Session()
    current = check_url(url, resolver)
    started = clock()
    for _ in range(MAX_REDIRECTS + 1):
        try:
            resp = session.get(current, stream=True, timeout=TIMEOUT_SECONDS, allow_redirects=False,
                               headers={'User-Agent': USER_AGENT, 'Accept': ', '.join(ACCEPTED)})
        except Exception as e:
            raise FetchError(f'the page could not be reached ({e.__class__.__name__})')
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                target = resp.headers.get('Location')
                if not target:
                    raise FetchError('the page redirected nowhere')
                if clock() - started > deadline_s:
                    raise FetchError('the page took too long to download')
                current = check_url(urljoin(current, target), resolver)
                continue
            if resp.status_code != 200:
                raise FetchError(f'the site answered with status {resp.status_code}')
            ctype = str(resp.headers.get('Content-Type') or '').split(';')[0].strip().lower()
            if ctype not in ACCEPTED:
                raise FetchError(f'a {ctype or "file of unknown type"} is not something that can be read')
            declared = resp.headers.get('Content-Length')
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError('the page is too large to read')
            chunks, size = [], 0
            for chunk in resp.iter_content(65536):
                if clock() - started > deadline_s:
                    raise FetchError('the page took too long to download')
                size += len(chunk)
                if size > max_bytes:
                    raise FetchError('the page is too large to read')
                chunks.append(chunk)
            data = b''.join(chunks)
            title = ''
            if ctype == 'application/pdf':
                text = _pdf_text(data)
            else:
                charset = 'utf-8'
                if 'charset=' in str(resp.headers.get('Content-Type') or '').lower():
                    charset = str(resp.headers['Content-Type']).lower().split('charset=')[1].split(';')[0].strip()
                try:
                    decoded = data.decode(charset, errors='replace')
                except LookupError:
                    decoded = data.decode('utf-8', errors='replace')
                if ctype == 'text/plain':
                    text = decoded
                else:
                    page = html_to_text(decoded)
                    text, title = page['text'], page['title']
            text = text[:MAX_TEXT_CHARS]
            if len(text.strip()) < MIN_TEXT_CHARS:
                raise FetchError('the page has too little readable text (it may need a login or be built by scripts)')
            return {'url': current, 'title': title, 'text': text, 'content_type': ctype}
        finally:
            close = getattr(resp, 'close', None)
            if close:
                close()
    raise FetchError('the page redirected too many times')
