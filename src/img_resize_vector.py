from PIL import Image
import os
import numpy as np
import shutil


def resize_and_convert_images(input_dir, output_dir, output_vector_file):
    all_vectors = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(image_path).convert('RGB')
                new_image = image.resize((56, 56))
                new_image.save(os.path.join(output_dir, filename))


                image_array = np.array(new_image, dtype=np.float64) / 255.0
                vector = image_array.flatten()
                all_vectors.append(vector)
            except Exception as e:
                print(f"Error processing {filename}: ", e)

    all_vectors = np.array(all_vectors)
    np.save(output_vector_file, all_vectors)
    print(f"Les vecteurs ont été sauvegardés dans {output_vector_file}")


input_dirs = [
    "C:/projet_annuel/src/img/img_scraping/marguerite_fleur/",
    "C:/projet_annuel/src/img/img_scraping/rose_rouge/",
    "C:/projet_annuel/src/img/img_scraping/tulipe_jaune/"
]
output_dirs = [
    "C:/projet_annuel/src/img/img_resized/marguerite_fleur_resized/",
    "C:/projet_annuel/src/img/img_resized/rose_rouge_resized/",
    "C:/projet_annuel/src/img/img_resized/tulipe_jaune_resized/"
]
output_vector_files = [
    "C:/projet_annuel/src/vector_data/marguerite_fleur_vectors.npy",
    "C:/projet_annuel/src/vector_data/rose_rouge_vectors.npy",
    "C:/projet_annuel/src/vector_data/tulipe_jaune_vectors.npy"
]

for output_dir in output_dirs:
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

for input_dir, output_dir, output_vector_file in zip(input_dirs, output_dirs, output_vector_files):
    resize_and_convert_images(input_dir, output_dir, output_vector_file)
