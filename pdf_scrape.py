import os
from typing import List
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
import json
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
AZ_STANDARDS = os.path.join(BASE_DIR, 'az-standards')


def main(standard_url) -> None:
    url = standard_url
    folder_location = os.path.join(AZ_STANDARDS, url.split('/')[-1])
    links: List[str] = []

    # Create pdf downloads folder if it doesn't exist
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)

    # Get all pdf links from the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.select("a[href$='.pdf']"):
        links.append(link['href'])

    for link in soup.find_all('a', string='pdf', href=lambda href: href and 'GetDocumentFile' in href):
        url = link['href']
        response = requests.get(url)
        links.append(response.url.split('?id=')[0])

    # Download each pdf to the folder location
    for link in tqdm(links):
        filename = link.split('/')[-1]
        log.info(f"Downloading {filename} from {link}")
        with open(os.path.join(folder_location, filename), 'wb') as f:
            f.write(requests.get(link).content)

    # Save links to json file
    with open('links.json', 'w') as f:
        json.dump(links, f)


if __name__ == '__main__':
    math_standards = "https://www.azed.gov/standards-practices/k-12standards/mathematics-standards"
    science_standards = "https://www.azed.gov/standards-practices/k-12standards/standards-science"
    main(math_standards)
    main(science_standards)