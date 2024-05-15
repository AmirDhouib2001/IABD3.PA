import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Liste des classes de recherche (tasses, pot à stylo, souris d'ordinateur)
query_classes = ['tulipe jaune', 'rose rouge', 'marguerite fleur']
num_images_per_class = 2000


driver = webdriver.Firefox()

# Fonction pour collecter et enregistrer des images pour une classe donnée
def collect_and_save_images_for_class(query_class, num_images, start_index):
    output_directory = f'dataset_images/{query_class}'
    os.makedirs(output_directory, exist_ok=True)

    # Ouvrez Bing Images
    driver.get("https://www.bing.com/images/feed?form=Z9LH")

    search_box = driver.find_element(By.NAME, "q")
    search_box.clear()
    search_box.send_keys(query_class)
    search_box.send_keys(Keys.RETURN)

    time.sleep(3)

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    images = driver.find_elements(By.TAG_NAME, "img")

    for i, image in enumerate(images[:num_images]):
        img_url = image.get_attribute("src")
        if img_url:
            img_name = f"{i + start_index}.jpg"
            img_path = os.path.join(output_directory, img_name)
            urllib.request.urlretrieve(img_url, img_path)
            print(f"Téléchargement de l'image {i + start_index} de {query_class} terminé.")

# Démarrez l'index à 400
start_index = 0

for query_class in query_classes:
    collect_and_save_images_for_class(query_class, num_images_per_class, start_index)
    start_index += num_images_per_class

# Fermez le navigateur
driver.quit()

print("Collecte et enregistrement terminés.")
