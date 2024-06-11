import os
import time
import urllib.request
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")  # Si vous voulez exécuter Chrome en mode headless
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

fleurs = [
    {"type": "marguerite_fleur", "query": "marguerite", "path": "C:/projet_annuel/src/img/marguerite_fleur"},
    {"type": "rose_rouge", "query": "rose rouge", "path": "C:/projet_annuel/src/img/rose_rouge"},
    {"type": "tulipe_jaune", "query": "tulipe jaune", "path": "C:/projet_annuel/src/img/tulipe_jaune"},
]

urls = {
    "google": "https://www.google.com/imghp?hl=en",
    "getty": "https://www.gettyimages.fr",
    "unsplash": "https://www.unsplash.com/",
    "bing": "https://www.bing.com/images/feed?form=Z9LH",
    "adobe": "https://stock.adobe.com/fr",
    "pexels": "https://www.pexels.com/fr-fr/",
}

def get_input_xpath(key):
    if key == "google":
        return "//input[@name='q']"
    elif key == "adobe":
        return "//input[@name='search_term']"
    elif key == "pexels":
        return "//input[@name='search_query']"
    elif key == "bing":
        return "//input[@name='q']"
    elif key == "unsplash":
        return "//input[@placeholder='Search free high-resolution photos']"
    elif key == "getty":
        return "//input[@placeholder='Search']"


def download_images(site, query, path):
    driver.get(urls.get(site))

    try:
        input_element = driver.find_element(By.XPATH, get_input_xpath(site))
        input_element.send_keys(query)
        input_element.send_keys(Keys.RETURN)
    except Exception as e:
        logging.error(f"Error finding search input on {site}: {e}")
        return

    time.sleep(5)

    if site != "google":
        images = driver.find_elements(By.TAG_NAME, "img")
    else:
        images = driver.find_elements(By.CSS_SELECTOR, ".rg_i")

    if not os.path.exists(path):
        os.makedirs(path)

    existing_files = set(os.listdir(path))
    existing_urls = set()

    for file in existing_files:
        with open(os.path.join(path, file), 'rb') as img_file:
            existing_urls.add(img_file.read())

    nb_files = len(existing_files)

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            scroll_attempts += 1
            if scroll_attempts >= 3:
                break
        else:
            scroll_attempts = 0
            last_height = new_height

    for index, image in enumerate(images):
        try:
            img_url = image.get_attribute("src")
            if img_url is None:
                img_url = image.get_attribute("data-src")
            if img_url is None:
                continue
            img_name = f"{site.capitalize()}_{query.replace(' ', '_')}_{nb_files + index + 1}.jpg"
            img_path = os.path.join(path, img_name)
            img_content = urllib.request.urlopen(img_url).read()
            if img_content in existing_urls:
                print(f"Image {img_name} déjà téléchargée")
                continue
            with open(img_path, 'wb') as img_file:
                img_file.write(img_content)
            print(f"Image {img_name} téléchargée")
            existing_urls.add(img_content)
        except Exception as e:
            logging.info(f"Couldn't retrieve this image because {e}")
            continue

    logging.info(f"{query} images downloaded to {path}")

for fleur in fleurs:
    print(f"Téléchargement des images pour {fleur['query']} dans {fleur['path']}")
    download_images("bing", fleur["query"], fleur["path"])

driver.quit()

