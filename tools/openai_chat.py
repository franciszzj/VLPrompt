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
MODEL = "gpt-3.5-turbo"


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
def get_pair_level_description(subject, object):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are asked to play the role of a relation guesser. Given the category names of two objects in an image, you are to infer what kind of relation might exist between them based on your knowledge, and provide the reasons for each possible relation. In the relation between the two objects in the image, we refer to one object as the subject and the other as the object. There may or may not be a relation between the subject and the object. Please note that this relation has an order, that is, the subject comes first and the object comes after. If there is a relation between the two, these relations must belong to one of the pre-defined 56 different types."},
            {"role": "assistant",
             "content": "What are the 56 relations?"},
            {"role": "user",
             "content": "They are 'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing', 'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on'."},
            {"role": "assistant", "content": "Can you give me an example?"},
            {"role": "user", "content": "For example, the subject is a person, and the object is a sports ball. The possible relations between them could be:\n1. Beside: The person could be standing beside the sports ball.\n2. Looking at: The person might be looking at the ball to better control it.\n3. Playing: This is because it's very common in real life for a person to be playing with a sports ball.\n4. Chasing: The person might be chasing after the ball."},
            {"role": "assistant",
                "content": "Ok, I got it. Please give me the subject and object of the image."},
            {"role": "user", "content": "The subject is a {}, and the object is a {}.".format(
                subject, object)},
        ],
        temperature=0,
        n=1,
    )
    # logger.info(response["choices"][0]["message"]["content"])
    # return response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"], response["choices"][0]["message"]["content"]
    return response.usage.prompt_tokens, response.usage.completion_tokens, response.choices[0].message.content


@retry(tries=3, delay=1)
def get_triplet_level_description(subject, object, relation):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are asked to play the role of a relation judger. Given the category names of two objects in an image, and providing you with a relation category name, you need to predict whether this relation is likely to exist in the image based on your knowledge, and give the reason for its existence. For two objects, we call the first object subject and the second object object."},
            {"role": "assistant",
             "content": "Yes, I understand. Can you give me an example?"},
            {"role": "user",
             "content": "For example, the input is: the subject is a 'person', the object is a 'sports ball' and the relation is 'playing'. The output should be Yes, the relation is likely to exist in the image. This is because it's very common in real life for a person to be playing with a sports ball."},
            {"role": "assistant",
             "content": "Ok, I got it. Please give me the subject, object and relation names."},
            {"role": "user",
             "content": "The subject is a {}, the object is a {}, and the relation is {}".format(
                 subject, object, relation)},
        ],
        temperature=0,
        n=1,
    )
    # logger.info(response["choices"][0]["message"]["content"])
    # return response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"], response["choices"][0]["message"]["content"]
    return response.usage.prompt_tokens, response.usage.completion_tokens, response.choices[0].message.content


def process(db_dir, bin=6, part=0):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    kv_db = dbm.open(os.path.join(db_dir, 'kv.db'), 'c')
    input_token_count = 0
    output_token_count = 0
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
            logger.info("speed: {:.4f}, left: {:.2f}s, {}: {}/133, {}: {}/133, input_token_count: {}, output_token_count: {}".format(
                speed, left, subject, i, object, j, input_token_count, output_token_count))
            t0 = time.time()

            #################
            # Triplet-level #
            #################
            for relation in relation_categories:
                # Check if it has been executed
                key = subject + '#' + object + '#' + relation
                if key in kv_db:
                    continue
                # Get triplet-level description
                try:
                    input_token_num, output_token_num, description = get_triplet_level_description(
                        subject, object, relation)
                    input_token_count += input_token_num
                    output_token_count += output_token_num
                    kv_db[key] = pickle.dumps(description)
                except Exception as e:
                    logger.warn(e)
                    continue

            ##############
            # Pair-level #
            ##############
            # Check if it has been executed
            key = subject + '#' + object
            if key in kv_db:
                continue
            # Get pair-level description
            try:
                input_token_num, output_token_num, description = get_pair_level_description(
                    subject, object)
                input_token_count += input_token_num
                output_token_count += output_token_num
                kv_db[key] = pickle.dumps(description)
            except Exception as e:
                logger.warn(e)
                continue

    kv_db.close()


if __name__ == '__main__':
    db_dir = sys.argv[1]
    bin = int(float(sys.argv[2]))
    part = int(float(sys.argv[3]))
    process(db_dir, bin, part)
