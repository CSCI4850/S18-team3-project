''' 
Scrapes transcripts of television shows from websites.
Sources:
    Rick and Morty seasons 1-3: rickandmorty.wikia.com/wiki/Category:Season_x_transcripts
'''

import requests
import re
import os
from tqdm import tqdm
from time import sleep
from bs4 import BeautifulSoup as bs


def get_rick_and_morty():
    # will replace @ with the season number while running
    root_url = "http://rickandmorty.wikia.com"
    base_url = root_url + "/wiki/Category:Season_@_transcripts"

    for season in [1,2,3]:
        cur_url = base_url.replace('@', str(season))
        page = requests.get(cur_url)
        print("Pinged {} with status code {}".format(cur_url, page.status_code))

        # create soup and look at html tags
        soup = bs(page.content, 'html.parser')

        # get all episode links
        episodes = []
        for link in soup.findAll('a', attrs={'href': re.compile("/Transcript")}):
            episodes.append(root_url + link.get('href'))

        # save text to file
        filepath = os.path.join("data", "train", "rick_and_morty.txt")
        with open(filepath, 'a',encoding='utf8') as f:
            for episode in episodes:
                episode_page = requests.get(episode)
                episode_soup = bs(episode_page.content, 'html.parser')
                text = episode_soup.findAll('div', attrs={'class': 'poem'})
                for script in text:
                    if len(script) > 0:
                        f.write(script.get_text())


if __name__ == "__main__":
    get_rick_and_morty()
    print("complete")
