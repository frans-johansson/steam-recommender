import re


def id_from_url(url):
    """
    Tries to extract the application ID from a Steam store URL.
    May return `None` if the regular expression search fails.
    """
    found = re.search('app\/(\d+)', url)
    if found:
        return found.group(1)
    return None
