import json
from json import load
import numpy as np
from PIL import Image
from io import BytesIO
import io
import asyncio
import sys
import logging


logging.basicConfig(
    format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)



#------------------------------------------------------------
def read_json_data(file_name) :
    with open(file_name , encoding="utf-8") as json_file:
        temp = json.load(json_file)
    return(temp)

#------------------------------------------------------------
def write_json_data(data, file_name) :
    #print(str(data))
    with open(f'{file_name}', 'w', encoding="utf-8") as outfile:
        json.dump(data , outfile, indent=4, default=lambda o: o.__dict__)
        print(f'JSON data saved in : {file_name}')



#------------------------------------------------------------
def load_file(path):
    with open(path, "r" , encoding="utf-8") as io_file:
        return io_file.read()


#------------------------------------------------------------
def load_json_file(path):
    with open(path, "r", encoding="utf-8") as json_string:
        return load(json_string)



#------------------------------------------------------------
def get_bytes_from_a_incomming_photo(message):
    photo_size = message.photo
    telegram_file = photo_size.get_file()
    byte = telegram_file.download_as_bytearray()
    return byte


#------------------------------------------------------------
def write_file(data, file_path):
    with open(file_path, 'wb' , encoding="utf-8") as f:
        f.write(data)


#------------------------------------------------------------
def float_from_user_input(user_input):
    return float(
        user_input
        .replace(" ", "")
        .replace("\n", "")
        .replace(",", ".")
    )
#------------------------------------------------------------
def integer_from_user_input(user_input):
    return int(float_from_user_input(user_input))

#------------------------------------------------------------
def get_key(dictionary, value):
    """ Function to return the key that match with the passed value
        >>> get_key({'py' : 3.14, 'other' : 666}, 3.14)
        'py'
    """
    values = list(dictionary.values())
    keys = list(dictionary.keys())
    return (keys[
        values.index(value)
        ])

#------------------------------------------------------------
def markdownize(words):
    print('markdownize() ')
    return words.replace('.','\.').replace('-','\-').replace('!','\!').replace('|','\|').replace('#','\#').replace('=','\=').replace('+','\+').replace('(','\(').replace(')','\)').replace('~','\~').replace('_','\_')


#------------------------------------------------------------
def error_function(update, context):
    """Log Errors caused by Updates."""
    print(f'Error Log Function : {update} \n {context.error}')
    logger.warning(f'')


#------------------------------------------------------------
def convert_pil_img_to_sendable_bytes(pil_image: Image):
    bio = BytesIO()
    bio.name = f'temp.jpeg'
    pil_image.save(bio, 'JPEG')
    bio.seek(0)
    return bio



import cv2
#------------------------------------------------------------
def pil_image_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#------------------------------------------------------------
def cv_image_to_pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


#------------------------------------------------------------
def unpack_nested_list(listed_lists):
    return [val for sublist in listed_lists for val in sublist]
    

#------------------------------------------------------------
def progress(count, total, suffix=''):
    bar_len = 20
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s | %s/%s ...%s\r' % (bar, percents, '%', count, total, suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben
    if count == total: print(f"\n")
