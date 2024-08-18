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


from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from ..utils.utils import  read_json_data, write_json_data
from .ai_dataclasses 	import User
#------------------------------------------------------------
"""
"@dataclass
class User():
    id                      : int
    username                : str
    total_openai_calls      : int
    total_craiyon_calls     : int
    total_local_ai_calls    : int
    prompts                 : List = field(default_factory=lambda: [])

    def __init__(self ):
        self.id = 0
        self.username = ""
        self.total_openai_calls = 0
        self.total_craiyon_calls = 0
        self.prompts = []

    def __repr__(self):
        return str(self.__dict__)

    def first(self, effective_user):
        self.id                     = effective_user.id
        self.username               = effective_user.username
        self.total_openai_calls     = 0
        self.total_craiyon_calls    = 0
        self.total_local_ai_calls   = 0
        self.total_llm_calls        = 0
        self.prompts                = []
        return

    def from_json_dict(self, d):
        self.id                     = d['id']
        self.username               = d['username']
        self.total_openai_calls     = d['total_openai_calls']
        self.total_craiyon_calls    = d['total_craiyon_calls']
        self.prompts                = d['prompts']
        self.total_local_ai_calls   = d['total_local_ai_calls']
        self.total_llm_calls        = d['total_local_ai_calls']

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

    def add_local_ai_call(self, user_prompt):
        self.total_local_ai_calls += 1
        self.prompts.append(user_prompt)
        return

    def add_llm_call(self, user_prompt):
        self.total_llm_calls += 1
        self.prompts.append(user_prompt)
        return

    def get_api_totals(self):
        return {'openai': self.total_openai_calls , 'craiyon': self.total_craiyon_calls, 'stable_diffusion': self.total_local_ai_calls, 'prompts': self.prompts}

    def get_api_total_sum(self):
        return (self.total_openai_calls + self.total_craiyon_calls + self.total_local_ai_calls)

    def save_number(self):
        return self.total_craiyon_calls + 1
"""
##------------------------------------------------------------
def all_users_api_stats_callback(update, context):
    user_info(update, context)
    user_id = update.effective_user.id
    username =  update.effective_user.username
    api_leaderboard = {}
    bot_user_data = context.bot_data.copy()
    try:
        bot_user_data.pop('openai_settings')
    except:
        pass

    for user_id, users in bot_user_data.items():
        api_leaderboard.update({users.username: users.get_api_total_sum()})

    leaderboard = dict(sorted(api_leaderboard.items(),reverse=True, key=lambda item: item[1]))
    #userdata = userdata
    title = f"🔐  TOTAL API CALLS  🏆"
    text = f".{title: ^36}.\n\n"
    i=0
    for name,total in leaderboard.items():
        text += f"{total:>6}\t{name: ^30}\t\n"
        i += 1
        if i>5: break

    #text += f"="*20 + f"\n"
    total_users = f"(Total Users: {len(leaderboard)})"
    text += f"\n"
    text += f".{total_users: ^46}."
    update.message.reply_text(text)
    return

##------------------------------------------------------------
def api_stats_callback(update, context):
    user_info(update, context)
    user_id = update.effective_user.id
    username =  update.effective_user.username

    userdata = context.bot_data[user_id].get_api_totals()
    print(userdata)
    #userdata = userdata
    title = f"🔐 HISTORY OF API CALLS"
    usr = f"Username: {username}"
    text = f".{title: ^36}.\n"
    text += f".{usr: ^36}.\n\n"
    for model in userdata:
        model_value = userdata[model]
        if model == 'prompts': model_value = len(userdata[model])
        text += f"{model: <35}{model_value: <10}\n"
    update.message.reply_text(text)
    return


##------------------------------------------------------------
def userdata_file_path():
    data_path   = f'./Sanskrit/data/AI_api_userdata.json'
    data_dir    = Path(data_path)
    return  data_dir

##------------------------------------------------------------
def load_json_userdata(update, context):
    userdata = read_json_data(userdata_file_path())
    for entry in userdata :
        user = User()
        user.from_json_dict(userdata.get(entry))
        context.bot_data.update({int(entry) : user})
    return


##------------------------------------------------------------
def user_info(update, context):
    print('user_info || Refreshing lasted saved bot_data.')
    try:
        load_json_userdata(update, context)
    except:
        text = "no json file found"
    user_id = update.effective_user.id
    bot_users_ids = [str(id_) for id_ in list(context.bot_data.keys())]
    if str(user_id) not in bot_users_ids:
        print(f"... Adding new user ({user_id}) to bot_data.")
        user = User()
        user.first(update.effective_user)
        context.bot_data.setdefault(user_id, user)

        bot_user_data = context.bot_data.copy()
        try:
            bot_user_data.pop('openai_settings')
        except:
            pass
        write_json_data(bot_user_data, userdata_file_path())
    return



#------------------------------------------------------------
#------------------------------------------------------------
api_stats_leaderboard_command   = CommandHandler("api_stats_leaderboard", all_users_api_stats_callback, run_async=True)
api_stats_command               = CommandHandler("api_stats", api_stats_callback, run_async=True)

#------------------------------------------------------------
#------------------------------------------------------------
