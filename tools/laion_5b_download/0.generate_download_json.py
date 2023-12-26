import os
import json

from tqdm import tqdm

from clip_client import ClipClient


def search_json_download(save_root, text, search_image_num):
    save_json = os.path.join(save_root, text + ".json")

    if not os.path.exists(save_json):
        client = ClipClient(url="https://knn.laion.ai/knn-service",
                            indice_name="laion5B-H-14",
                            num_images=search_image_num)
        results = client.query(text=text)
        with open(save_json, 'w') as file_obj:
            json.dump(results, file_obj)


if __name__ == "__main__":
    '''
    search prompt url:https://knn.laion.ai/
    '''
    search_prompt_file_path = r'D:\laion_5b_download\1.txt'
    save_json_folder_path = r'D:\laion_5b_download\save_jsons'
    search_image_num = 1000

    if not os.path.exists(save_json_folder_path):
        os.makedirs(save_json_folder_path)

    with open(search_prompt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    print(f'total json num: {len(lines)}')

    for index, class_name in tqdm(enumerate(lines)):
        print(class_name)
        search_json_download(save_json_folder_path, class_name,
                             search_image_num)
