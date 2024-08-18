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


from typing import List, Dict

from ..utils.utils 		import read_json_data, write_json_data
from ..const.tokens 	import MODEL_ENGINE, ARBYBC_TELEGRAM_ID, SANSKRITTHEROBOT_TELEGRAM_ID


import cv2
import numpy as np
##------------------------------------------------------------

##------------------------------------------------------------

@dataclass
class ChatGPT():
    role_system			: 	str
    role_assistant		: 	str
    user_prompt 		: 	str
    last_response 		: 	str
    previous_prompt     :   str

    def default_system_role(self):
        return ""
        system_message  = f"ChatGPT, your name is 'Sanskrit The Robot', the robot fox."
        return system_message


    def default_assistant_role(self):
        assistant_message = f"Pull my tail! This magic foxy will answer yowr qwestions! OwO"
        return assistant_message

    def __init__(self):
        self.role_system        = self.default_system_role()
        self.role_assistant     = self.default_assistant_role()
        self.user_prompt        = ""
        self.last_response      = ""
        self.previous_prompt    = ""

    def from_json_dict(self, d, **kwargs):
        #print('... Restoring ChatGPT prompt settings ...')
        self.role_system        = d['role_system']
        self.role_assistant     = d['role_assistant']
        self.user_prompt        = d['user_prompt']
        self.last_response      = d['last_response']
        self.previous_prompt    = d['previous_prompt']
        return self



    def update_system_role(self, new_content):
        self.role_system    = new_content
        return

    def update_assistant_role(self, new_content):
        self.role_assistant    = new_content
        return

    def update_user_prompt_role(self, new_content):
        self.previous_prompt    = self.user_prompt
        self.user_prompt        = new_content
        return

    def update_last_response(self, new_content):
        self.last_response = new_content
        return

    def full_chatgpt_prompt(self):
        return [ {"role": "system", "content" : f'{self.role_system}'}, {"role": "assistant", "content" : f'{self.role_assistant}'}, {"role": "user", "content" : f'{self.user_prompt}'} ]


    def format_for_llama(self):
        system = f"<s>[INST] <<SYS>> \n\n" + f"{self.role_system} \n\n" + f"<</SYS>>\n\n </s><s> [/INST] "
        prev_prompt = f"[INST] " + f"{self.previous_prompt}" + f" [/INST] "
        prev_response = f"{self.last_response}" + f" </s><s> "
        prompt = f"[INST] " + f"{self.user_prompt}" + f" [/INST]"
        return ''.join([system, prev_prompt, prev_response, prompt])
##------------------------------------------------------------


#------------------------------------------------------------
@dataclass
class User():
    id                      : int
    username                : str
    total_openai_calls      : int
    total_craiyon_calls     : int
    total_local_ai_calls    : int
    prompts                 : List = field(default_factory=lambda: [])
    chatgpt                 : Dict = field(default_factory=lambda: {})

    def __init__(self ):
        self.id = 0
        self.username = ""
        self.total_openai_calls = 0
        self.total_craiyon_calls = 0
        self.prompts = []
        self.chatgpt                = ChatGPT()


    def __repr__(self):
        return str(self.__dict__)

    def first(self, effective_user):
        self.id                     = effective_user.id
        self.username               = effective_user.username
        self.total_openai_calls     = 0
        self.total_craiyon_calls    = 0
        self.total_sd_calls         = 0
        self.total_llm_calls        = 0
        self.prompts                = []
        return

    def from_json_dict(self, d):
        self.id                     = d['id']
        self.username               = d['username']
        self.total_openai_calls     = d['total_openai_calls']
        self.total_craiyon_calls    = d['total_craiyon_calls']
        self.prompts                = d['prompts']
        self.total_sd_calls         = d['total_sd_calls']
        self.total_llm_calls        = d['total_llm_calls']
        self.chatgpt = ChatGPT().from_json_dict(d['chatgpt'])
        return

    def add_prompt(self, user_prompt):
        self.prompts.append(user_prompt)
        return

    def add_craiyon_call(self, user_prompt):
        self.total_craiyon_calls += 1
        self.prompts.append(user_prompt)
        return

    def add_openai_call(self, user_prompt):
        self.total_openai_calls += 1
        self.prompts.append(user_prompt)
        return

    def add_stable_diffusion_call(self, user_prompt):
        self.total_sd_calls += 1
        self.prompts.append(user_prompt)
        return

    def add_llm_call(self, user_prompt):
        self.total_llm_calls += 1
        self.prompts.append(user_prompt)
        return

    def get_api_totals(self):
        return {'OpenAI': self.total_openai_calls , 'Craiyon': self.total_craiyon_calls, 'Stable Diffusion': self.total_sd_calls, 'Llama':self.total_llm_calls , 'prompts': self.prompts}

    def get_api_total_sum(self):
        return (self.total_openai_calls + self.total_craiyon_calls + self.total_sd_calls + self.total_llm_calls)

    def save_number(self):
        return (self.total_craiyon_calls + self.total_sd_calls )


##------------------------------------------------------------

@dataclass
class OpenAiSettings():
    api_key : str
    key_owner : int
    allow_usage : bool
    chance_to_bark : int
    image_resolution : str
    chatgpt             : ChatGPT()

    def __init__(self):
        self.api_key = f""
        self.key_owner = 12345
        self.allow_usage = True
        self.chance_to_bark = 1
        self.image_resolution = ""
        self.medium()
        self.chatgpt = ChatGPT()

    def get_api_key(self):
        return self.api_key


    def is_active(self):
        if self.api_key == "":
            return False
        return True

    def set_api_key(self, new_api_key):
        self.api_key = new_api_key

    def show_api_key(self):
        key_length = len(self.api_key)
        #print(f'{key_length}')
        censored_key = f"{self.api_key[:7]}" + '*'*(key_length-7-4) + f"{self.api_key[-4:]}"
        #print(f'{censored_key} \n length {len(censored_key)}')
        return censored_key

    def set_key_owner(self, owner_id):
        self.key_owner = owner_id


    def allow_global_usage(self):
        self.allow_usage = True

    def disallow_global_usage(self):
        self.allow_usage = False


    def flip_usage(self):
        if self.allow_usage :
            print(f'true to false')
            self.disallow_global_usage()
        else:
            print(f'false to true')
            self.allow_global_usage()


    def small(self):
        self.image_resolution = "256x256"

    def medium(self):
        self.image_resolution = "512x512"

    def large(self):
        self.image_resolution = "1024x1024"


    def set_chance_to_bark(self, percent):
        self.chance_to_bark = percent


    def file_path(self):
        return Path('Sanskrit/data/OpenAi_Settings.json')


    def bury_json_settings(self):
        data = self.__dict__
        write_json_data(data, self.file_path())


    def fetch_json_settings(self):
        try:
            d = read_json_data(self.file_path())
            print(f'Json data read...')
            self.api_key = d['api_key']
            self.key_owner = d['key_owner']
            self.allow_usage = d['allow_usage']
            self.chance_to_bark = d['chance_to_bark']
            self.image_resolution = d['image_resolution']
            self.chatgpt = ChatGPT().from_json_dict(d['chatgpt'])
            print(f'... settings fully restored.')
            return self

        except:
            print('no json found, writing first version.')
            self.bury_json_settings()
            return self

