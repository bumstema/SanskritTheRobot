import telegram
from telegram import Update, BotCommand
from telegram.ext import Updater, CallbackContext, Filters, MessageHandler,  ConversationHandler, CommandHandler, CallbackQueryHandler
from telegram import ReplyKeyboardRemove, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup


import os

from dataclasses import dataclass, field, asdict
from pathlib import Path

# Include Random Number Generator
import random
from random import randint
import requests

from nltk.tokenize import sent_tokenize
import nltk

from .api_stats 		import User, user_info, userdata_file_path
from ..utils.utils 		import read_json_data, write_json_data
from ..const.tokens 	import MODEL_ENGINE, ARBYBC_TELEGRAM_ID, SANSKRITTHEROBOT_TELEGRAM_ID



from typing import Optional
import fire
from llama_cpp import Llama
import json
import argparse
import subprocess


ckpt_dir = os.getcdw() + f'/llama-main/llama-2-7b-chat-bin/'
tokenizer_path = os.getcdw() + f'/llama-main/llama-2-7b-chat-bin/'
llama_bin_folder = os.getcdw() + f'/llama-main/llama-2-7b-chat-bin/'

def set_user_chatmodel_system_message(update, context):
    user_prompt = (' '.join(context.args)).strip()
    chatbot_settings = context.bot_data[update.effective_user.id]
    chatLLAMA = chatbot_settings.chatgpt
    chatLLAMA.update_system_role(f'{user_prompt}')
        # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())
    context.bot.send_message(chat_id=f'{update.effective_chat.id}', text=f'System Message for User updated to: {user_prompt}', reply_to_message_id = update.message.message_id)
    return

set_user_llm_system_msg_command = CommandHandler("set_llama", set_user_chatmodel_system_message )

##------------------------------------------------------------
def save_convo(update, context):
    print(f'Adding prompt to user: {update.effective_user.id} statistics.')

    user_info(update, context)
    user_id = update.effective_user.id
    context.bot_data[user_id].add_llm_call(context.bot_data[user_id].chatgpt.user_prompt)
    # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())
    return

##------------------------------------------------------------
def parse_llama_response(reply):
    text_output = reply.strip()
    print(f'1. {text_output = } \n')
    if '\x1c' in text_output: return f'🤭 Oh my! Should I?'
    if text_output == '': return f'🤭 Oh my! Should I?'
    #try:
    text_output = text_output.replace('SANSKRIT:','SYSTEM:').replace('SANSKRIT THE ROBOT:','SYSTEM:').split("SYSTEM:")
    print(f'2. {text_output = } \n {len(text_output) = } \n')
    for index, t in enumerate(text_output): print(f"  -{index} {t}\n")
    if "USER:" in text_output: text_output = [txt.split("USER:") for txt in text_output][0]
    print(f'3. {text_output = } \n {len(text_output) = } \n')
    #
    #print(f'4. {text_output = } \n {len(text_output) = } \n')
    text_output = random.sample(text_output, k=1)
    print(f'4. {text_output = } \n {len(text_output) = } \n')
    text_fold = " ".join(text_output)
    print(text_fold)
    print(str((("AI " in text_fold) or ("AI," in text_fold) or ("AI language" in text_fold))))
    if (("AI " in text_fold) or ("AI," in text_fold) or ("AI language" in text_fold)):
        text_output = text_fold.split("\n")
        if not isinstance(text_output, list):
            text_output = text_output.split("</s>")
            if not isinstance(text_output, list):
                text_output = [text_output]
        
        text_output = [ t for t in text_output if not (("AI " in t) or ("AI," in text_output) or ("AI language" in t)) ]
        
        if (("AI " in text_output) or ("AI," in text_output) or ("AI language" in text_output) or text_output == []):
            text_output = [f'😳 uhh.. maybe']
        return text_output[0]

    #except:
    #    text_output = f'😳 uhh.. maybe'
    if isinstance(text_output, list):
        if len(text_output) > 0: text_output = text_output[0]
    print(f'----< No reference to "AI language" detected.\n')
    text_output = text_output.replace("</s>","").replace("<s>","").replace("[INST]","").replace("[/INST]","")
    return text_output
##------------------------------------------------------------
def llama_main(update, context):
    llama_bin_folder =   os.getcdw() + f'/llama-main/llama-2-7b-chat-bin/llama-2-7b-chat.ggmlv3.q8_0.bin'
    llama_bin_folder =   os.getcdw() + f'/llama-main/llama-2-13b-chat-bin/llama-2-13b-chat.ggmlv3.q4_1.bin'

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=f"{llama_bin_folder}")
    args = parser.parse_args()
    LLM = Llama(model_path=args.model, n_ctx=1024, n_gpu_layers=1)



    try:
        user_prompt = (' '.join(context.args)).strip()
    except:
        user_prompt =  (update.effective_message.text).strip()

    print(f'Llama : {update.effective_user.username} in {update.effective_chat.title}')
    print(f'Prompt: {user_prompt}')
 
    user_info(update, context)
    user_id = update.effective_user.id

    msg = context.bot.send_message(chat_id=f'{update.effective_chat.id}', text=f'(...⏳)', reply_to_message_id = update.message.message_id)
    #try:
    chatbot_settings = context.bot_data[update.effective_user.id]
    chatLLAMA = chatbot_settings.chatgpt
    

    if chatLLAMA.role_system == "": chatLLAMA.update_system_role(defined_system_message())
    chatLLAMA.update_user_prompt_role(f'{user_prompt}')

    
    print(f'{context.bot_data = }')
    print(f'Sending to LLM...')
    print(f'{chatLLAMA.format_for_llama()}')
    output = LLM( chatLLAMA.format_for_llama(), max_tokens=0)
    
    print(f'{output}')
    text_output = output["choices"][0]["text"]
    
    text_output = parse_llama_response(text_output)
    chatLLAMA.update_last_response(f'{text_output}')
    context.bot.edit_message_text(f'{text_output}', chat_id=update.effective_chat.id, message_id=msg.message_id)
    user_id = update.effective_user.id
    context.bot_data[user_id].add_llm_call(context.bot_data[user_id].chatgpt.user_prompt)
    # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())
    #except:
    #context.bot.edit_message_text(f'😅 Whoops.. um,?', chat_id=update.effective_chat.id, message_id=msg.message_id)
    return

def defined_system_message():
    system_message  ="Your name is 'Sanskrit The Robot', a Fox.
    return system_message


def default_system_role(prompt):
    system_message  = f"<s>[INST] <<SYS>> Your name is 'Sanskrit The Robot', a Fox.
    system_message += f"<</SYS>> \n\n [/INST]"
    system_message += f"USER: {prompt} \n\n"
    return system_message


llama_convo_command = CommandHandler("llama", llama_main )

llama_convo_setbotcommands  = [ BotCommand( f'llama',f'..write me a prompt and I\'ll reply.')]
