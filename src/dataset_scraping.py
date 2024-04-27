import os
import time
import urllib
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys



# chercher les rose , marguerittes ,tulipes

driver = webdriver.Firefox()
query = "rose"
field_type = "rose"
path = f'./img/{field_type}'


urls = {
    "google": f"https://www.google.com/imghp?hl=en",
}

site = "google"
driver.get(urls.get(site))

def get_input_xpath(key):
    if key == "google":
        return "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea"

input_element = driver.find_element(By.XPATH, get_input_xpath(site))
input_element.send_keys(query)
input_element.send_keys(Keys.RETURN)

time.sleep(5)

if site != "google":
    images = driver.find_elements(By.TAG_NAME, "img")
else:
    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i")

nb_files = len(os.listdir(path))

for index, image in enumerate(images):
    try:
        img_url = image.get_attribute("src")
        img_name = f"{site.capitalize()}_{field_type}_{index + 1}.jpg"
        img_path = os.path.join(path, img_name)
        urllib.request.urlretrieve(img_url, img_path)
    except Exception as e:
        logging.log(logging.INFO, f"fail , because {e}")
        continue

logging.log(logging.INFO, f"{query} success : images downloaded")

driver.quit()
