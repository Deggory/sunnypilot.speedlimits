"""The comma pill: its address set from the site, used by the check and by the download."""
from __future__ import annotations

import pytest

from nnslr_tools import comma_link


def test_an_address_is_kept_and_an_empty_one_goes_back_to_the_ssh_host(tmp_path):
    assert comma_link.configured(tmp_path) is None
    assert comma_link.set_address(tmp_path, ' 192.168.1.20 ') == '192.168.1.20'
    assert comma_link.configured(tmp_path) == '192.168.1.20'
    assert comma_link.set_address(tmp_path, 'comma@comma.local') == 'comma@comma.local'
    assert comma_link.set_address(tmp_path, '') is None
    assert comma_link.configured(tmp_path) is None


@pytest.mark.parametrize('address', ['192.168.1.20; rm -rf /', '-oProxyCommand=x', 'a b', 'host:22', '$(id)', 'x' * 300])
def test_anything_but_a_host_is_refused(tmp_path, address):
    with pytest.raises(comma_link.AddressError):
        comma_link.set_address(tmp_path, address)
    assert comma_link.configured(tmp_path) is None


def test_the_last_login_that_worked_is_remembered(tmp_path, monkeypatch):
    # A comma switched off after it once took this server's key is not offered the setup from zero again.
    assert comma_link.last_login(tmp_path) is None
    monkeypatch.setattr(comma_link, '_check', lambda: True)
    monkeypatch.setattr(comma_link, '_login', lambda: False)
    monkeypatch.setattr(comma_link, 'state', lambda root: {'last_login': comma_link.last_login(root)})
    assert comma_link.check_now(tmp_path) == {'last_login': None}
    monkeypatch.setattr(comma_link, '_login', lambda: True)
    seen = comma_link.check_now(tmp_path)['last_login']
    assert seen and seen.endswith('+00:00')
    # Away now: the check fails, the time of the last login stays.
    monkeypatch.setattr(comma_link, '_check', lambda: False)
    assert comma_link.check_now(tmp_path)['last_login'] == seen
    comma_link.seen_path(tmp_path).write_text('not json')
    assert comma_link.last_login(tmp_path) is None
