"""Following a link from an uploaded document (src/utils/safe_fetch.py): the
address comes from a document someone else wrote, so anything that is not a
public https page is refused, and a hostile redirect cannot get round that."""
import pytest

from src.utils import safe_fetch as sf

PUBLIC = lambda host: ['93.184.216.34']
PAGE = '<html><head><title>Acme results</title><style>p{}</style></head><body><script>steal()</script>' \
       '<p>' + 'Acme grew profit 18 percent to KES 7.4 billion. ' * 8 + '</p></body></html>'


class Resp:
    def __init__(self, status=200, body=b'', ctype='text/html; charset=utf-8', headers=None):
        self.status_code, self.body = status, body
        self.headers = {'Content-Type': ctype, **(headers or {})}
        self.closed = False

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, *responses):
        self.responses, self.urls = list(responses), []

    def get(self, url, **kw):
        assert kw['allow_redirects'] is False and kw['stream'] is True and kw['timeout'] <= 20
        self.urls.append(url)
        return self.responses.pop(0)


@pytest.mark.parametrize('url,why', [
    ('http://example.com/a', 'only https'),
    ('ftp://example.com/a', 'only https'),
    ('file:///etc/passwd', 'only https'),
    ('https://user:pw@example.com/a', 'password'),
    ('https://example.com:8443/a', 'standard https port'),
    ('https:///nohost', 'no address'),
    ('', 'only https'),
])
def test_unsafe_shapes_are_refused(url, why):
    with pytest.raises(sf.FetchError, match=why):
        sf.check_url(url, PUBLIC)


@pytest.mark.parametrize('ip', ['127.0.0.1', '10.0.0.5', '192.168.1.1', '172.16.0.9', '169.254.169.254',
                                '0.0.0.0', '::1', 'fe80::1', 'fc00::1', '::ffff:127.0.0.1', '100.64.0.1', '224.0.0.1'])
def test_a_name_that_points_inside_is_refused(ip):
    with pytest.raises(sf.FetchError, match='private or internal'):
        sf.check_url('https://looks-innocent.example/a', lambda h: [ip])


def test_one_private_address_among_public_ones_is_enough_to_refuse():
    with pytest.raises(sf.FetchError, match='private or internal'):
        sf.check_url('https://x.example/a', lambda h: ['93.184.216.34', '10.0.0.1'])


def test_an_address_that_cannot_be_found_is_refused_plainly():
    def nowhere(host):
        raise sf.FetchError(f'the address {host} could not be found (gaierror)')
    with pytest.raises(sf.FetchError, match='could not be found'):
        sf.check_url('https://nope.example/a', nowhere)
    assert sf.check_url('https://example.com/a?b=1', PUBLIC) == 'https://example.com/a?b=1'


def test_a_page_is_read_as_text_without_its_scripts_or_styles():
    out = sf.fetch('https://example.com/r', Session(Resp(body=PAGE.encode())), PUBLIC)
    assert out['title'] == 'Acme results' and 'profit 18 percent' in out['text']
    assert 'steal' not in out['text'] and 'p{}' not in out['text'] and out['content_type'] == 'text/html'


def test_a_redirect_is_followed_by_hand_and_every_hop_is_checked():
    seen = []

    def resolver(host):
        seen.append(host)
        return ['10.0.0.9'] if host == 'internal.example' else ['93.184.216.34']

    hostile = Session(Resp(301, headers={'Location': 'https://internal.example/secret'}))
    with pytest.raises(sf.FetchError, match='private or internal'):
        sf.fetch('https://public.example/a', hostile, resolver)
    assert hostile.urls == ['https://public.example/a'] and seen == ['public.example', 'internal.example']
    ok = Session(Resp(302, headers={'Location': '/final'}), Resp(body=PAGE.encode()))
    assert sf.fetch('https://public.example/a', ok, PUBLIC)['url'] == 'https://public.example/final'


def test_a_redirect_to_plain_http_or_in_a_loop_is_refused():
    with pytest.raises(sf.FetchError, match='only https'):
        sf.fetch('https://a.example/x', Session(Resp(302, headers={'Location': 'http://a.example/y'})), PUBLIC)
    loop = Session(*[Resp(302, headers={'Location': '/again'}) for _ in range(6)])
    with pytest.raises(sf.FetchError, match='too many times'):
        sf.fetch('https://a.example/x', loop, PUBLIC)


def test_size_type_and_status_limits():
    with pytest.raises(sf.FetchError, match='too large'):
        sf.fetch('https://a.example/x', Session(Resp(body=b'x' * 5000)), PUBLIC, max_bytes=1000)
    with pytest.raises(sf.FetchError, match='too large'):
        sf.fetch('https://a.example/x', Session(Resp(body=b'x', headers={'Content-Length': '99999999'})), PUBLIC)
    with pytest.raises(sf.FetchError, match='not something that can be read'):
        sf.fetch('https://a.example/x', Session(Resp(body=b'MZ', ctype='application/x-msdownload')), PUBLIC)
    with pytest.raises(sf.FetchError, match='status 404'):
        sf.fetch('https://a.example/x', Session(Resp(404)), PUBLIC)
    with pytest.raises(sf.FetchError, match='too little readable text'):
        sf.fetch('https://a.example/x', Session(Resp(body=b'<html><script>app()</script></html>')), PUBLIC)


def test_a_connection_failure_is_reported_in_plain_words_and_the_response_is_closed():
    class Boom:
        def get(self, *a, **k):
            raise ConnectionError('refused')
    with pytest.raises(sf.FetchError, match='could not be reached'):
        sf.fetch('https://a.example/x', Boom(), PUBLIC)
    resp = Resp(body=PAGE.encode())
    sf.fetch('https://a.example/x', Session(resp), PUBLIC)
    assert resp.closed


def test_plain_text_is_accepted():
    text = ('Dividend of KES 1.50 per share was declared. ' * 8).encode()
    out = sf.fetch('https://a.example/x.txt', Session(Resp(body=text, ctype='text/plain')), PUBLIC)
    assert 'KES 1.50' in out['text']
