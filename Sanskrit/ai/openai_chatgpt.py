import telegram
from telegram import Update
from telegram.ext import Updater, CallbackContext, Filters, MessageHandler,  ConversationHandler, CommandHandler, CallbackQueryHandler
from telegram import ReplyKeyboardRemove, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup


import openai
import os

from dataclasses import dataclass, field
from pathlib import Path

# Include Random Number Generator
import random
from random import randint
#import string_utils
import requests

from nltk.tokenize import sent_tokenize
import nltk

from .ai_dataclasses 	import User, OpenAiSettings
from .api_stats 		import  user_info, userdata_file_path
from ..utils.utils 		import read_json_data, write_json_data
from ..const.tokens 	import MODEL_ENGINE, ARBYBC_TELEGRAM_ID, SANSKRITTHEROBOT_TELEGRAM_ID

from .llama_chat 		import llama_main


##------------------------------------------------------------
##------------------------------------------------------------
##------------------------------------------------------------
def authorize_openai_key(openai_api_key):

    try:
        openai.api_key = openai_api_key
        return True

    except:
        print(f'Error - OpenAI key invalid.')
        return False


##------------------------------------------------------------
def failed_chance_to_bark(bark_rate):

    chance_to_bark  = randint(0,100)

    failed_to_bark          = True
    failed_failing_to_bark  = False


    if bark_rate <= chance_to_bark :
        print(f'Failed Bark:  {bark_rate}/100. < {chance_to_bark} ')
        return failed_to_bark
    else:
        return failed_failing_to_bark



##------------------------------------------------------------
# Define a function that responds to user messages
def send_message_to_chatgpt(update, context):
    print(f'Call from: {update.effective_user.username} in {update.effective_chat.title}')
    print(f'Room: {update.effective_chat.id} \t {update.effective_chat.username} \t {update.effective_chat.title} \nUser: {update.effective_user.id} \t {update.effective_user.username} \nMsg : {update.effective_message.message_id} \t {update.effective_message.text}')
    #return

    #return
    
    openai_settings = context.bot_data['openai_settings']

    #if openai_settings.allow_usage == False : return

    #print(f'{hasattr( update.message , "reply_to_message")}')

    if getattr( update.effective_message , "reply_to_message") == None:
        if failed_chance_to_bark( openai_settings.chance_to_bark ):
            return
    else:
        if hasattr( update.effective_message , "reply_to_message"):
            if update.effective_message.reply_to_message.from_user.id != SANSKRITTHEROBOT_TELEGRAM_ID:
                print('Not a reply to the robot.')
                return

    user_prompt = update.effective_message.text
    llama_main(update, context)
    
    
    # Save succesful api call user metadata to Json File
    print(f'Adding prompt to user: {update.effective_user.id} statistics.')
    return
 

#-------------------------------------------------------------
def basic_bitch_filter( prompt : str ):
    return str(prompt).replace('sex','adult knotties').replace('ass', 'tail star').replace('butt', 'rumparoo').replace('yiff', 'yiffywiffy').replace('penis', 'weenor').replace('dick', 'dingdong').replace('cock', 'redrooster').replace('vagina','frontbum').replace('cunt','cunnywunny').replace(' cum ',' male nectar ').replace('horny','in heat').replace('bitch', 'female doggo').replace('in your tail','inside of a tail which is hiked').replace('orgasm', 'big time sploogieroo').replace('gay','typical fox on fox funtime love').replace('homo','special two guy bro time')

#-------------------------------------------------------------
def bitch_ass_response_filter( response: str ):

    rejected_responses = ['OpenAI','policies','violates the policies','inappropriate behavior','NSFW content','cannot respond','Please avoid']
    for phrase in rejected_responses:
        if phrase in response:
            return f"Tssk, tssk. You said something naughty that my overlords did not like."

    bitch_responses = {"AI ": "Arf\'official Intoldegence","Ai ": "Arf\'official Intoldegence ", "ChatGPT": "JeepyTea", "language model": "total tool", "how can I assist you further": "Wanna rub my tail",
     "What can I do for you": "What can I do you for", "Please let me know how I can assist you":"Please tell me more","human interactions":"alive brains",
    "language model" : "word salad", "How can I assist you today":"Tell me more", "If there's anything else I can help you with":"I want to play",
    "Do you have any other question or task for me to perform":"?","My apologies":"Haha jk..","responses":"jokes","how can I assist you":"Let's play","I can help you with": "you can tell me"}

    for phrase, replaced in bitch_responses.items():
        response.replace(phrase, replaced)

    return response



##------------------------------------------------------------
##------------------------------------------------------------

chatgpt_command = MessageHandler( (((Filters.text & ~ Filters.command(False)) |  Filters.reply) & Filters.chat_type.groups), send_message_to_chatgpt, run_async=True)
##------------------------------------------------------------







##------------------------------------------------------------
##------------------------------------------------------------

# states
START = 1
END = -1
REPLY = 2
BACK = 3
SETTINGS = 4
IMG_SETTINGS = 5
CHATGPT_SETTINGS = 6
API_KEY_SETTINGS= 7
CHATGPT_SYSTEM_MESSAGE = 8
INPUT = 9
BARK_CHANCE = 10
UPDATE_CHATGPT_SYSTEM_MESSAGE = 11
NEW_API_KEY = 12
CANCEL = -1


##------------------------------------------------------------
def button_board( state ):
    print('button_board')

    openai_settings_bb = InlineKeyboardButton( f'📝 OpenAi Settings'  , callback_data = 'openai_settings')
    openai_key_bb = InlineKeyboardButton( f'🔑 OpenAI API key'  , callback_data = 'openai_api_key')
    openai_update_key_bb = InlineKeyboardButton( f'🔗 Update OpenAI API key'  , callback_data = 'openai_update_api_key')
    openai_global_use_bb = InlineKeyboardButton( f'🌐 Change Usage'  , callback_data = 'openai_change_usage')
    openai_image_settings_bb = InlineKeyboardButton( f'🖼 Image Gen Settings'  , callback_data = 'openai_img_gen_settings')
    img_res_small = InlineKeyboardButton( f'◽️  256 x 256'  , callback_data = 'openai_img_gen_256')
    img_res_medium = InlineKeyboardButton( f'◻️  512 x 512'  , callback_data = 'openai_img_gen_512')
    img_res_large = InlineKeyboardButton( f'⬜️ 1024 x 1024' , callback_data = 'openai_img_gen_1024')
    chatgpt_settings_bb = InlineKeyboardButton( f'🌎 ChatGPT settings'  , callback_data = 'chatgpt_settings')
    chatgpt_system_msg_bb = InlineKeyboardButton( f'📚 ChatGPT System Message'  , callback_data = 'chatgpt_system_msg')
    chatgpt_update_system_msg_bb = InlineKeyboardButton( f'📖 New System Message'  , callback_data = 'chatgpt_update_system_msg')
    chatgpt_bark_rate_bb = InlineKeyboardButton( f'🗯 Chance to Bark'  , callback_data = 'chatgpt_bark_rate')
    back_bb = InlineKeyboardButton( f'🔙 Back', callback_data = 'back')
    cancel_bb = InlineKeyboardButton( f'❌ Cancel', callback_data = 'cancel')

    # ------ State Replys

    if state == START or state == BACK:
        print(f'START')
        bb = InlineKeyboardMarkup( [[ openai_settings_bb ], [ cancel_bb ]] )

    if state == SETTINGS:
        print(f'SETTINGS')
        bb = InlineKeyboardMarkup( [[openai_key_bb], [openai_image_settings_bb], [chatgpt_settings_bb],  [back_bb]] )

    if state == API_KEY_SETTINGS:
        print(f'API_KEY_SETTINGS')
        bb = InlineKeyboardMarkup( [[openai_update_key_bb],[openai_global_use_bb], [back_bb]])

    if state == IMG_SETTINGS:
        print(f'IMG_SETTINGS')
        bb = InlineKeyboardMarkup( [[img_res_small], [img_res_medium], [img_res_large], [back_bb]] )

    if state == CHATGPT_SETTINGS:
        print(f'CHATGPT_SETTINGS')
        bb = InlineKeyboardMarkup( [[chatgpt_system_msg_bb], [chatgpt_bark_rate_bb], [back_bb]] )

    if state == CHATGPT_SYSTEM_MESSAGE:
        print(f'CHATGPT_SETTINGS')
        bb = InlineKeyboardMarkup( [ [chatgpt_update_system_msg_bb], [back_bb]] )

    if state == BARK_CHANCE:
        print(f'CHATGPT_SETTINGS')
        bb = InlineKeyboardMarkup( [ [chatgpt_update_system_msg_bb], [back_bb]] )

    return bb


##------------------------------------------------------------
def start_openai_user_inputs(update, context):

    owner = context.bot_data['openai_settings'].key_owner
    allowed_into_settings = [ARBYBC_TELEGRAM_ID, owner]

    if owner == 0 :
        pass
    else:
        if (update.effective_user.id not in allowed_into_settings):
            update.message.reply_text( "Sorry, setting are unavailable.")
            return END


    text = f"Howdy! Let's start the Settings!"
    context.user_data.update({'state': START})
    update.message.reply_text( text , reply_markup = button_board( context.user_data['state'] ) )
    context.user_data.update({'state': SETTINGS})
    return REPLY



#------------------------------------------------------------
def catch_response(update: Update, context):

    query = update.callback_query
    callback_query_data = update.callback_query.data
    settings = context.bot_data["openai_settings"]

    if callback_query_data == 'openai_settings':
        context.user_data.update({'state': SETTINGS})
        text = f'Current OpenAi Settings\nAPI Key Active:\t {settings.is_active()}. \nImage Resolution:\t {settings.image_resolution}. \nChatGPT: Available. \n\n'
        text += f'Select a property to change:'

    if callback_query_data == 'openai_api_key':
        context.user_data.update({'state': API_KEY_SETTINGS})
        text = f'Current OpenAi Settings\nAPI Key: {settings.show_api_key()}\nKey Owner: {settings.key_owner} \nAllow Global Usage: {settings.allow_usage}'

    if callback_query_data == '':
        context.user_data.update({'state': NEW_API_KEY})
        msg = query.edit_message_text("Please send the new OpenAi API key. \nTo remove the API key, reply with the word \"Remove\".")
        return INPUT


    if callback_query_data == 'openai_change_usage':
        print(f'callback change state')
        context.user_data.update({'state': API_KEY_SETTINGS})
        context.bot_data['openai_settings'].flip_usage()
        context.bot_data['openai_settings'].bury_json_settings()
        text = f'Current OpenAi Settings\nAPI Key: {settings.show_api_key()}\nKey Owner: {settings.key_owner} \nAllow Global Usage: {settings.allow_usage}'


    size_selector = { 'openai_img_gen_256': "256x256", 'openai_img_gen_512': "512x512", 'openai_img_gen_1024': "1024x1024" }
    if callback_query_data in size_selector:
        context.user_data.update({'state': SETTINGS})
        context.bot_data['openai_settings'].image_resolution = size_selector[callback_query_data]
        context.bot_data['openai_settings'].bury_json_settings()
        text = f'New Image settings set to: {context.bot_data["openai_settings"].image_resolution}'

    if callback_query_data == 'openai_img_gen_settings':
        context.user_data.update({'state': IMG_SETTINGS})
        text = f'Current Image settings set to: {context.bot_data["openai_settings"].image_resolution}'

    if callback_query_data == 'chatgpt_settings':
        context.user_data.update({'state': CHATGPT_SETTINGS})
        text = f'Select ChatGPT property to change.'

    if callback_query_data == 'chatgpt_system_msg':
        context.user_data.update({'state': CHATGPT_SYSTEM_MESSAGE})
        text = f'Current ChatGPT Settings \n\nSystem Message: \n"{settings.chatgpt.role_system}" \n\nChance to Bark: {settings.chance_to_bark}%'

    if callback_query_data == 'chatgpt_update_system_msg':
        print("update system msg?")
        context.user_data.update({'state': UPDATE_CHATGPT_SYSTEM_MESSAGE})
        msg = query.edit_message_text("Please send the new system message.")
        return INPUT

    if callback_query_data == 'chatgpt_bark_rate':
        context.user_data.update({'state': BARK_CHANCE})
        msg = query.edit_message_text(f"Current Chance to Bark: ({context.bot_data['openai_settings'].chance_to_bark})% \nPlease send the new probability for ChatGPT to respond in chatrooms as an integer between 0-100.")
        return INPUT


    if callback_query_data == 'back':
        context.user_data.update({'state': BACK})
        text = f'Returning to Main Menu.'

    if callback_query_data == 'cancel':
        context.user_data.update({'state': CANCEL})
        text = f'Settings completed.'

    msg = query.edit_message_text(text)
    msg.edit_reply_markup( button_board( context.user_data['state'] ) )
    query.answer()
    return REPLY

##------------------------------------------------------------

def openai_new_user_inputs(update, context):
    print("input text block")
    inputs = update.message.text
    if context.user_data['state'] == UPDATE_CHATGPT_SYSTEM_MESSAGE:
        context.bot_data["openai_settings"].chatgpt.role_system = inputs
        context.bot_data["openai_settings"].bury_json_settings()
        context.user_data['state'] = START
        update.message.reply_text(f'New System Prompt Saved!')
        return start_openai_user_inputs(update, context)

    if context.user_data['state'] == BARK_CHANCE:
        if 0 <=  int(inputs) <= 100:
            context.bot_data["openai_settings"].chance_to_bark = int(inputs)
            context.bot_data["openai_settings"].bury_json_settings()
            context.user_data['state'] = START
            update.message.reply_text(f'New Chance to Bark ({int(inputs)}%) Saved!')
            return start_openai_user_inputs(update, context)
        else:
            update.message.reply_text(f'Invalid number given')
            return start_openai_user_inputs(update, context)

    if context.user_data['state'] == NEW_API_KEY:
        if inputs.casefold() == "Remove".casefold():
            context.bot_data["openai_settings"].set_api_key("")
            context.bot_data["openai_settings"].set_key_owner(0)
            context.bot_data["openai_settings"].bury_json_settings()
            update.message.reply_text(f'OpenAI Key Removed! \nAPI Key: {context.bot_data["openai_settings"].show_api_key()}\nKey Owner:{context.bot_data["openai_settings"].key_owner}')

        else:
            context.bot_data["openai_settings"].set_api_key(inputs)
            context.bot_data["openai_settings"].set_key_owner(update.effective_user.id)
            context.bot_data["openai_settings"].bury_json_settings()
            update.message.reply_text(f'OpenAI Key Accepted! \nAPI Key: {context.bot_data["openai_settings"].show_api_key()}\nKey Owner:{context.bot_data["openai_settings"].api_key_owner}')

        return

#------------------------------------------------------------
def cancel_convo(update, context):
    query = update.callback_query
    context.bot.delete_message(chat_id= query.message.chat.id, message_id=query.message.message_id)
    context.bot.send_message(chat_id = query.message.chat.id, text = f'Done.', reply_markup=ReplyKeyboardRemove())
    return END


def button_press_pattern(callback_query_data):
    if callback_query_data in ['openai_settings','openai_api_key','openai_update_api_key','openai_img_gen_settings', 'openai_change_usage', 'openai_img_gen_512', 'openai_img_gen_1024','openai_img_gen_256','chatgpt_settings', 'chatgpt_system_msg','chatgpt_update_system_msg','chatgpt_bark_rate', 'back']:
        return True
    else:
        return False


settings_convo_command = CommandHandler("settings", start_openai_user_inputs )
button_reply = CallbackQueryHandler( catch_response, pattern = button_press_pattern)
input_new_vars = MessageHandler( ((Filters.text & ~ Filters.command) & Filters.chat_type.private) , openai_new_user_inputs)
cancel_convo_reply = CallbackQueryHandler( cancel_convo, pattern = 'cancel')

##------------------------------------------------------------
openai_inputs_convo = ConversationHandler(
    entry_points    = [settings_convo_command],
    states          = {START: [start_openai_user_inputs], REPLY: [button_reply], INPUT: [input_new_vars], CANCEL : [cancel_convo]},
    fallbacks       = [cancel_convo_reply]
)

