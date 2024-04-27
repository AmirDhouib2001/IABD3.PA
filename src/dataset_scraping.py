import os
import time
import urllib
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
field_type = "ward"
query = "rose"
path = f'./img/{field_type}'

urls = {
    "google": f"https://www.google.com/imghp?hl=en",
    "getty": "https://www.gettyimages.fr",
    "unsplash": "https://www.unsplash.com/",
    "bing": "https://www.bing.com/images/feed?form=Z9LH",
    "adobe": "https://stock.adobe.com/fr",
    "pexels": "https://www.pexels.com/fr-fr/",
}
site = "google"
driver.get(urls.get(site))


def get_input_xpath(key):
    if key == "google":
        return "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea"
    elif key == "adobe":
        return "/html/body/div/div/div/main/section/div[1]/section/div[1]/div[3]/div/div[2]/div/div/input"
    elif key == "pexels":
        return "/html/body/div[2]/header/div/form/div/input"
    if key == "bing":
        return "/html/body/header/form/div/input[1]"
    elif key == "unsplash":
        return "/html/body/div/div/header/nav/div[2]/form/div[1]/input"
    elif key == "getty":
        return "/html/body/div[2]/section/div/div[1]/div/div[2]/div/div/div/div/div/div[1]/div[1]/form/input"


if site == "google":
    auto_reject_cookies_button = driver.find_element(By.ID, "W0wltc")
    auto_reject_cookies_button.click()
elif site == "getty":
    auto_reject_cookies_button = driver.find_element(By.CSS_SELECTOR, "#onetrust-accept-btn-handler")
    auto_reject_cookies_button.click()

input_element = driver.find_element(By.XPATH, get_input_xpath(site))
input_element.send_keys(query)
input_element.send_keys(Keys.RETURN)

if site == "pexels":
    time.sleep(7)
else:
    time.sleep(5)

if site != "google":
    images = driver.find_elements(By.TAG_NAME, "img")
else:
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i")

nb_files = len(os.listdir(path))

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

for index, image in enumerate(images):
    try:
        img_url = image.get_attribute("src")
        img_name = f"{site.capitalize()}_{query.split(' ')[-1]}_{index + 1}.jpg"
        img_path = os.path.join(path, img_name)
        urllib.request.urlretrieve(img_url, img_path)
    except Exception as e:
        logging.log(logging.INFO, f"Couldn't retrieve this image because {e}")
        continue

logging.log(logging.INFO, f"{query} images downloaded")

driver.quit()
