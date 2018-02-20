''' 
Scrapes transcripts of television shows from websites.
Sources:
    Rick and Morty seasons 1-3: rickandmorty.wikia.com/wiki/Category:Season_x_transcripts
'''

import requests
from bs4 import BeautifulSoup as bs

########## First get R&M scripts for testing ###########

# will replace @ with the season number while running
base_url = "http://rickandmorty.wikia.com/wiki/Category:Season_@_transcripts"

for season in [1,2,3]:
    if season != 1: continue
    cur_url = base_url.replace('@', str(season))
    page = requests.get(cur_url)
    print("Pinged {} with status code {}".format(cur_url, page.status_code))

    # create soup and look at html tags
    soup = bs(page.content, 'html.parser')

    for text in soup.find_all('div'):
        if "This article" in text.get_text()[:20]:
            print(text.get_text())

