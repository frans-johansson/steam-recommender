import re
from steam_api.api_requests import get_game_data
from bs4 import BeautifulSoup


def id_from_url(url):
    """
    Tries to extract the application ID from a Steam store URL.
    May return `None` if the regular expression search fails.
    """
    found = re.search('app\/(\d+)', url)
    if found:
        return found.group(1)
    return None


def clean_game_description(id):
    """
    Given a game ID, this function returns a cleaned version of its detailed description from the Steam store.
    The description is assumed to be in HTML format. Hyperlinks are removed, and a full stop is used to separate the text content of the various tags.
    """
    try:
        desc = get_game_data(str(id))['detailed_description']
    except KeyError as e:
        print(e)
        return ''

    soup = BeautifulSoup(desc, 'html.parser')

    # Remove <a> tags with URLs
    for a in soup.find_all('a', href=True):
        a.extract()

    return soup.get_text(separator='. ')
