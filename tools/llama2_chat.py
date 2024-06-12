import os
import sys
import dbm
import time
import pickle
from loguru import logger
from llama import Llama
from retry import retry

# Config
# llama2_dir = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/work_dirs/checkpoints/llama2'
llama2_dir = '/scratch/grp/grv_shi/k21163430/work_dirs/checkpoints/llama2'
ckpt_dir = '{}/llama-2-7b-chat'.format(llama2_dir)
tokenizer_path = '{}/tokenizer.model'.format(llama2_dir)
max_seq_len = 1024
max_batch_size = 4
temperature = 0.6
top_p = 0.9
max_gen_len = None

# llama2 chat generator
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


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
    dialogs = [
        [
            {"role": "system", "content": "You are asked to play the role of a relation guesser. Given the category names of two objects in an image, you are to infer what kind of relation might exist between them based on your knowledge, and provide the reasons for each possible relation. In the relation between the two objects in the image, we refer to one object as the subject and the other as the object. There may or may not be a relation between the subject and the object. Please note that this relation has an order, that is, the subject comes first and the object comes after. If there is a relation between the two, these relations must belong to one of the pre-defined 56 different types. What are the 56 relations?"},
            {"role": "user", "content": "They are 'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off', 'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on', 'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying', 'looking at', 'guiding', 'kissing', 'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on'."},
            {"role": "assistant", "content": "Can you give me an example?"},
            {"role": "user", "content": "For example, the subject is a person, and the object is a sports ball. The possible relations between them could be:\n1. Beside: The person could be standing beside the sports ball.\n2. Looking at: The person might be looking at the ball to better control it.\n3. Playing: This is because it's very common in real life for a person to be playing with a sports ball.\n4. Chasing: The person might be chasing after the ball."},
            {"role": "assistant",
                "content": "Ok, I got it. Please give me the subject and object of the image."},
            {"role": "user", "content": "The subject is a {}, and the object is a {}.".format(
                subject, object)},
        ]
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # logger.info(result[0]['generation']['content'])
    return results[0]['generation']['content']


@retry(tries=3, delay=1)
def get_triplet_level_description(subject, object, relation):
    dialogs = [
        [
            {"role": "system", "content": "You are asked to play the role of a relation judger. Given the category names of two objects in an image, and providing you with a relation category name, you need to predict whether this relation is likely to exist in the image based on your knowledge, and give the reason for its existence. For two objects, we call the first object subject and the second object object. Please give me an example."},
            {"role": "user", "content": "For example, the input is: the subject is a 'person', the object is a 'sports ball' and the relation is 'playing'. The output should be Yes, the relation is likely to exist in the image. This is because it's very common in real life for a person to be playing with a sports ball."},
            {"role": "assistant",
                "content": "Ok, I got it. Please give me the subject, object and relation names."},
            {"role": "user", "content": "The subject is a {}, the object is a {}, and the relation is {}".format(
                subject, object, relation)},
        ]
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    # logger.info(result[0]['generation']['content'])
    return results[0]['generation']['content']


def process(db_dir, bin=36, part=0):
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    kv_db = dbm.open(os.path.join(db_dir, 'kv.db'), 'c')
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
            # Triplet-level #
            #################
            for relation in relation_categories:
                # Check if it has been executed
                key = subject + '#' + object + '#' + relation
                if key in kv_db:
                    continue
                # Get triplet-level description
                try:
                    description = get_triplet_level_description(
                        subject, object, relation)
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
                description = get_pair_level_description(
                    subject, object)
                kv_db[key] = pickle.dumps(description)
            except Exception as e:
                logger.warn(e)
                continue

    kv_db.close()


if __name__ == '__main__':
    # torchrun --nproc_per_node 1 tools/llama2_chat.py data/psg/meta/llama2-7b/ 1 0
    db_dir = sys.argv[1]
    bin = int(float(sys.argv[2]))
    part = int(float(sys.argv[3]))
    process(db_dir, bin, part)
