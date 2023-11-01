# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Include Telegram Bot Library (Python Telegram Bot)
### https://github.com/python-telegram-bot/
import telegram
from telegram import Update, Chat, ForceReply, BotCommand, ReplyKeyboardRemove, InputMediaPhoto
from telegram import InlineQueryResultArticle, ParseMode, InputTextMessageContent
from telegram import KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup

# Include text detection
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler
from telegram.ext import InlineQueryHandler, CallbackQueryHandler, CallbackContext, PicklePersistence, Defaults

# From Inline Commands and From Inline Keyboards
from telegram.utils.request import Request

from telegram.utils.helpers import escape_markdown
from telegram.error import NetworkError, Unauthorized
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


from craiyon import Craiyon, CraiyonV1, craiyon_utils

from PIL import Image
from io import BytesIO
import base64


# Include Debug and System Tools
import traceback
import sys, os
from pathlib import Path

from datetime import datetime, timedelta, time, date
from dataclasses import dataclass, field
from typing import List

from PIL import Image, ImageFilter, ImageOps, ImageChops

import json
import requests

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

from .api_stats import User, user_info, userdata_file_path
from ..utils.utils import read_json_data, write_json_data, convert_pil_img_to_sendable_bytes



#------------------------------------------------------------
#  Craiyon
#------------------------------------------------------------



# Craiyon class: which creates images using craiyon
class Call_Craiyon:
    def __init__(self, prompt):
        self.prompt = prompt

    # generates and saves the images
    def generate(self):
        craiyon = "https://api.craiyon.com/v3"
        payload = {
            "prompt": self.prompt,
        }
        try:
            response = requests.post(url=craiyon, json=payload, timeout=160)
            print(f'{response = }')
            result = response.json()["images"]
            return result
            for index, image in enumerate(result, start=1):
                with open(f"generated/{self.prompt}_{index}.webp", "wb") as file:
                    file.write(base64.decodebytes(image.encode("utf-8")))

        except Exception as ex:
            return False



##------------------------------------------------------------
def file_path(update, context):
    user_id = update.effective_user.id
    return f'./Sanskrit/data/ai_generations/craiyon_{user_id}_{context.bot_data[user_id].save_number()}.jpeg'



##------------------------------------------------------------
def send_solution_callback(update, context):
    try:
        file_name = file_path(update, context)
        update.message.reply_photo(open(file_name, 'rb'), caption = f'Last received image.')
    except:
        update.message.reply_text('Oopsies! Cannot open image file.')
    return




##------------------------------------------------------------
##------------------------------------------------------------
def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    #print('append_images()')
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im



##------------------------------------------------------------
def nine_to_one_image_reduction(craiyon_images):
    #print('nine_to_one_image_reduction()')
    #open_imgs = list(map(Image.open, craiyon_images))
    open_imgs = craiyon_images

    #print(f'{len(open_imgs[0:3])}')
    x = append_images(open_imgs[0:3], direction='horizontal')
    y = append_images(open_imgs[3:6], direction='horizontal')
    z = append_images(open_imgs[6:9], direction='horizontal')
    final = append_images([x,y,z], direction='vertical')
    #final.show()
    #final.save(file_path(update, context))
    #final = convert_pil_image_to_byte_array(final)


    return final





##------------------------------------------------------------
def craiyon_callback(update, context):
    print('craiyon_callback()')

    user_info(update, context)
    user_id = update.effective_user.id
    print(f'Call from: {update.effective_user.username} in {update.effective_chat.title}')
    print(context.args)
    user_prompt = ' '.join(context.args)
    print(user_prompt)
    print(f'{context.args = }\n{user_prompt = }')

    msg = context.bot.send_photo(f'{update.effective_chat.id}', open('./Sanskrit/ai/ai_photos/craiyon.png','rb'), caption = f'Sending prompt: \"{user_prompt}\" to Craiyon. ⏳' , reply_to_message_id = update.message.message_id)

    return
    print('Generating Images using Craiyon API.')
    #generator = Craiyon()
    #result = generator.generate(f'{user_prompt}', negative_prompt="cap", model_type="art")
    #images = result.images

    generator = Call_Craiyon(f'{user_prompt}')
    result = generator.generate()
    images = result

    print('Images Received Successfully.')

    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open('./Sanskrit/ai/ai_photos/craiyon_inverted.png','rb'), caption = f'Successful Craiyon creation! Now sending to chat ... ⌛️'), timeout = 60 )


    print('Adding prompt to user statistics.')
    context.bot_data[user_id].add_craiyon_call(user_prompt)
    # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())
    write_json_data(context.bot_data, userdata_file_path())


    print('Number of images returned: '+str(len(images)))




    images = craiyon_utils.encode_base64(result.images)
    craiyon_images = [ Image.open(BytesIO(base64.decodebytes(i))).convert("RGB") for i in images]



    print(f'Images ({len(craiyon_images)}) decoded by BytesIO.')

    single_image = nine_to_one_image_reduction(craiyon_images)
    print(f'Photo Concatination from ({len(craiyon_images)}) to 1')


    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( convert_pil_img_to_sendable_bytes(single_image), caption = f'Craiyon Prompt: \"{user_prompt}\"'), timeout = 60  )

    print(f'Sent to Telegram: {update.effective_chat.title}')
    return
    context.bot.edit_message_caption(chat_id = f'{update.effective_chat.id}',  message_id = msg.message_id, caption = f'Oopsies! Generation Failed 🚮')

    print('...failed')
    return



#------------------------------------------------------------
#------------------------------------------------------------
craiyon_command = CommandHandler("craiyon", craiyon_callback, run_async=True)
crayion_solution_command = CommandHandler("send_solution", send_solution_callback)
craiyon_setbotcommands  = [ BotCommand( 'craiyon','Prompt a picture from Craiyon!')]

#------------------------------------------------------------
#------------------------------------------------------------

