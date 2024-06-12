import os
import sys
import dbm
import time
import pickle
from loguru import logger
from openai import OpenAI
from retry import retry

# OpenAI API key
client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY", "PUT_YOUR_KEY"))


def replace_name(text):
    if '-stuff' in text:
        text = text.replace('-stuff', '')
    if '-merged' in text:
        text = text.replace('-merged', '')
    if '-other' in text:
        text = text.replace('-other', '')
    return text


object_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                     'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
object_categories = [replace_name(x) for x in object_categories]
relation_categories = ['over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing',
                       'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']


@retry(tries=3, delay=1)
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def process(text_dir, embed_dir, bin=1, part=0):
    if not os.path.exists(text_dir):
        logger.error("text_dir: {} not exists".format(text_dir))
        return
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    text_kv_db = dbm.open(os.path.join(text_dir, 'kv.db'), 'r')
    embed_kv_db = dbm.open(os.path.join(embed_dir, 'kv.db'), 'c')

    t0 = time.time()
    for i, subject in enumerate(object_categories):
        if not (i >= (part * len(object_categories) // bin) and i < ((part + 1) * len(object_categories) // bin)):
            continue
        for j, object in enumerate(object_categories):
            # Time and log
            t1 = time.time()
            speed = 1 / (t1 - t0)
            left = ((len(object_categories) - i - 1) * len(object_categories) // bin +
                    len(object_categories) - j - 1) * (t1 - t0)
            logger.info("speed: {:.4f}, left: {:.2f}s, {}: {}/133, {}: {}/133".format(
                speed, left, subject, i, object, j))
            t0 = time.time()

            #################
            # Triplet level #
            #################
            for relation in relation_categories:
                # Check if it has been executed
                key = subject + '#' + object + '#' + relation
                if key in embed_kv_db:
                    continue
                # Get text embed
                try:
                    text = pickle.loads(text_kv_db[key])
                    embed = get_embedding(text)
                    embed_kv_db[key] = pickle.dumps(embed)
                except Exception as e:
                    logger.warn(e)
                    continue

            ##############
            # Pair level #
            ##############
            # Check if it has been executed
            key = subject + '#' + object
            if key in embed_kv_db:
                continue
            # Get text embed
            try:
                text = pickle.loads(text_kv_db[key])
                embed = get_embedding(text)
                embed_kv_db[key] = pickle.dumps(embed)
            except Exception as e:
                logger.warn(e)
                continue

    text_kv_db.close()
    embed_kv_db.close()


if __name__ == '__main__':
    text_dir = sys.argv[1]
    embed_dir = sys.argv[2]
    bin = int(float(sys.argv[3]))
    part = int(float(sys.argv[4]))
    process(text_dir, embed_dir, bin, part)
