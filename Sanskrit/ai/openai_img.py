# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Include Telegram Bot Library (Python Telegram Bot)
### https://github.com/python-telegram-bot/
import telegram
from telegram import Update, Chat, ForceReply, BotCommand, ReplyKeyboardRemove, InputMediaPhoto, Message
from telegram import InlineQueryResultArticle, ParseMode, InputTextMessageContent
from telegram import KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup

# Include text detection
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler
from telegram.ext import InlineQueryHandler, CallbackQueryHandler, CallbackContext, PicklePersistence, Defaults

# From Inline Commands and From Inline Keyboards
from telegram.utils.request import Request

from telegram.utils.helpers import escape_markdown
from telegram.error import NetworkError, Unauthorized

#from .instruct_pix2pix import instruct_pix2pix
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#------------------------------------------------------------
#  OpenAI
#------------------------------------------------------------
import openai
import requests
from io import BytesIO
import base64
from PIL import Image, ImageFilter, ImageOps, ImageChops

from .api_stats 		import User, user_info, userdata_file_path
from ..utils.utils 		import read_json_data, write_json_data, convert_pil_img_to_sendable_bytes
from ..const.tokens 	import OPENAI_API_TOKEN
#------------------------------------------------------------



##------------------------------------------------------------
def authorize_openai_key(api_key):
    try:
        openai.api_key = api_key
        return True

    except:
        print(f'Error - OpenAI key invalid.')
        return False
    return

##------------------------------------------------------------
def image_from_prompt(user_prompt, resolution):
    print(f'image_from_prompt({resolution})')
    try:
        #image_size = "1024x1024"
        #image_size = "512x512"
        #image_size = "256x256"
        image_size = resolution
        response = openai.Image.create(prompt = user_prompt, n=1, size = image_size)
        print(f'Successful openai prompt sent.')
        image_url = response['data'][0]['url']
        return image_url
        #openai_response = requests.get(image_url)
        #img = BytesIO(openai_response.content)
        #print(f'Image successfully returned.')
        #return img
    except:
        error_text = 'Error: OpenAI API.'
        update.message.reply_text(error_text)
        return



##------------------------------------------------------------
def game_file_path(update):
    return  f'openai/{update.effective_message.chat.id}'


##------------------------------------------------------------
def save_image(img, game_name, game_type):
    print('save_image()')
    with Image.open(img) as img_file:
        img_file.save(f'{game_name}_{game_type}.jpg')
    return

##------------------------------------------------------------
def colour_image_to_find_edges(img, game_name):
    print('colour_image_to_find_edges()')
    with Image.open(img) as colour_img:
        image = colour_img
        #image.save(f'{game_name}_game.jpg')
        image.format = 'JPG'
        image = ImageOps.grayscale(image)
        image = image.filter(ImageFilter.FIND_EDGES)
        image = ImageOps.invert(image)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.6, percent=150, threshold=2))
        image = image.filter(ImageFilter.SMOOTH_MORE)
        image = ImageOps.autocontrast(image, preserve_tone=True)
        image.save(f'{game_name}_edge.jpg')
        #save_image(edge_img, game_name, "edge")
        edge_img = image
    return edge_img







##------------------------------------------------------------
def openai_callback(update, context):
    print('openai_callback()')
    return

    if not authorize_openai_key(context.bot_data['openai_settings'].api_key):
        update.message.reply_text(f'OpenAI Key Rejected. Please add a valid key in DMs.')
        return

    #openai.api_key = OPENAI_API_TOKEN


    try:
        user_info(update, context)
        user_id = update.effective_user.id


        print(context.args)
        user_prompt = ' '.join(context.args)
        print(user_prompt)
        file_path = game_file_path(update)


        msg = context.bot.send_photo(f'{update.effective_chat.id}', open('./Sanskrit/ai/ai_photos/openai.png','rb'), caption = f'Sending prompt: \"{user_prompt}\" to OpenAI. ⏳')

        #img = image_from_prompt(user_prompt)
        resolution = context.bot_data['openai_settings'].image_resolution
        img_url = image_from_prompt(user_prompt, resolution)


        context.bot_data[user_id].add_openai_call(user_prompt)

        # -- Remove settings and write ONLY user data to Json File
        bot_user_data = context.bot_data.copy()
        bot_user_data.pop('openai_settings')
        write_json_data(bot_user_data, userdata_file_path())
        #write_json_data(context.bot_data, userdata_file_path())

        #save_image(img, file_path, "openai")



        context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open('./Sanskrit/ai/ai_photos/openai-inverted.png','rb'), caption = f'Successful OpenAI creation! Now sending to chat ... ⌛️'), timeout = 60 )
        #context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( bio, caption = f'OpenAI Prompt: \"{user_prompt}\"'), timeout = 60 )
        try:
            print(f'Sending with url link.')
            context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( img_url, caption = f'OpenAI Prompt: \"{user_prompt}\"'), timeout = 60 )
            return
        except:
            print(f'Pil image to bytes')
            pil_img = Image.open(img)
            bio = BytesIO()
            bio.name = f'temp.jpeg'
            pil_image.save(bio, 'JPEG')
            bio.seek(0)
            context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open( bio, 'rb'), caption = f'OpenAI Prompt: \"{user_prompt}\" (bio)'), timeout = 60 )
            return

        #context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto(open(f'{game_file_path(update)}_openai.jpeg', 'rb'), caption = f'OpenAI Prompt: \"{user_prompt}\"'), timeout = 60 )

        #update.message.reply_photo(img, caption = f'Generated using prompt: \"{user_prompt}\"', reply_markup=ReplyKeyboardRemove())

        return

        edge_img = colour_image_to_find_edges(img, file_path)
        #save_image(edge_img, file_path, "edge")
        print('sending message : '+ str(edge_img))

        buttons = buttonboard()
        print(str(buttons))

        msg = update.message.reply_photo( open(f'{file_path}_edge.jpg', 'rb') , caption = f'Generated using prompt: \"{user_prompt}\". \n\n✨Find out if you more creative than a robot!🖍 Colour this page, then resend it as a reply to this message!', reply_markup = buttons)

        print(str(msg))
        context.chat_data.update({'msg' : msg})
        context.chat_data.update({'img' : img})
        context.chat_data.update({'edge_img' : edge_img})
        print(str(context.chat_data))

    except:
        context.bot.delete_message(chat_id= update.effective_chat.id, message_id=msg.message_id)
        update.message.reply_text("Please try again with an acceptable prompt!", reply_markup=ReplyKeyboardRemove())

    return



##------------------------------------------------------------
def send_solution_callback(update, context):
    try:
        game_name = game_file_path(update)
        update.message.reply_photo(open(f'{game_name}_game.jpg', 'rb'))
    except:
        update.message.reply_text('Oopsies! Something went wrong.')

    return
#------------------------------------------------------------
def get_photo_from_message(update, context):
    game_name = game_file_path(update)
    photo = update.message.photo
    print(f"{photo = }")
    photo = [item for item in photo if item.height == 256]
    photo = photo[0].get_file()
    photo_file_id = photo.file_id
    photo.download(f'{game_name}_{update.effective_user.id}.jpg')
    return photo


#------------------------------------------------------------
def photo_difference_score_minus_edges(update, context):
    game_name = game_file_path(update)
    with Image.open(f'{game_name}_game.jpg') as i1:
        with Image.open(f'{game_name}_{update.effective_user.id}.jpg') as i2:
            with Image.open(f'{game_name}_edge.jpg') as i3:

    #i1 = Image.open(f'{game_name}_game.jpg')
    #i2 = Image.open(f'{game_name}_{update.effective_user.id}.jpg')
    #i3 = Image.open(f'{game_name}_edges.jpg')
    #assert i1.mode == i2.mode, "Different kinds of images."
    #assert i1.size == i2.size, "Different sizes."
                edge_as_colour = ImageOps.colorize(i3, black ="black", white ="white", mid=None, blackpoint=0, whitepoint=255, midpoint=127)
                user_photo_without_edges = ImageChops.multiply(edge_as_colour,i2)
                #user_photo_without_edges.show()
                inverted_edge = ImageChops.invert(edge_as_colour)
                game_photo_with_edges = ImageChops.multiply(edge_as_colour,i1)
                #game_photo_with_edges.show()
                i1 = game_photo_with_edges
                i2 = user_photo_without_edges
                pairs = zip(i1.getdata(), i2.getdata())
                if len(i1.getbands()) == 1:
                    # for gray-scale jpegs
                    dif = sum(abs(p1-p2) for p1,p2 in pairs)
                else:
                    dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))

                ncomponents = i1.size[0] * i1.size[1] * 3
                value = (dif / 255.0 * 100) / ncomponents
                print ("Difference (percentage):", (dif / 255.0 * 100) / ncomponents)
                update.message.reply_text(f"That image is: {100.0 -value:.2f}% similar.")


#------------------------------------------------------------
"""
def photo_difference_score(update, context):
    game_name = game_file_path(update)
    i1 = Image.open(f'{game_name}_game.jpg')
    i2 = Image.open(f'{game_name}_{update.effective_user.id}.jpg')
    i3 = Image.open(f'{game_name}_edges.jpg')
    #assert i1.mode == i2.mode, "Different kinds of images."
    #assert i1.size == i2.size, "Different sizes."

    pairs = zip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1-p2) for p1,p2 in pairs)
    else:
        dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))

    ncomponents = i1.size[0] * i1.size[1] * 3
    value = (dif / 255.0 * 100) / ncomponents
    print ("Difference (percentage):", (dif / 255.0 * 100) / ncomponents)
    update.message.reply_text(f"That image is: {value:.2f}% similar.")

"""
#------------------------------------------------------------
def rate_coloured_image(update, context):
    print(str(update.effective_message.reply_to_message))
    coloured_photo = get_photo_from_message(update, context)
    photo_difference_score_minus_edges(update, context)
    #photo_difference_score(update, context)
    return






#------------------------------------------------------------
##------------------------------------------------------------
#------------------------------------------------------------
def get_photo_from_message(update, context):
    print(f'Getting photo from message...')
    photo = update.message.photo
    photo = [item for item in photo if item.height >= 512]
    photo = photo[0].get_file()
    photo_file_id = photo.file_id


    byte_array_photo = photo.download_as_bytearray()
    decoded_sent_photo = BytesIO(byte_array_photo)
    return decoded_sent_photo




#------------------------------------------------------------
def find_edges_of_colour_image_callback(update, context):

    sent_photo = get_photo_from_message(update, context)

    with Image.open(sent_photo) as colour_img:
        image = colour_img
        image.format = 'JPG'
        image = ImageOps.grayscale(image)
        image = image.filter(ImageFilter.FIND_EDGES)
        image = ImageOps.invert(image)
        image = image.filter(ImageFilter.UnsharpMask(radius=0.6, percent=150, threshold=2))
        image = image.filter(ImageFilter.SMOOTH_MORE)
        image = ImageOps.autocontrast(image, preserve_tone=True)

        bio = BytesIO()
        bio.name = f'temp.jpeg'
        image.save(bio, 'JPEG')
        bio.seek(0)
        context.bot.send_photo(f'{update.effective_chat.id}', photo=bio, caption=f'Edges detected.')


#------------------------------------------------------------
#------------------------------------------------------------
image_edge_command = MessageHandler(Filters.photo & Filters.caption_regex(r'outline'), find_edges_of_colour_image_callback, run_async=True)
#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------
#------------------------------------------------------------
openai_command = CommandHandler("openai", openai_callback, run_async=True)
openai_setbotcommands  = [ BotCommand( 'openai','Prompt an image with OpenAi')]
#------------------------------------------------------------
#------------------------------------------------------------
