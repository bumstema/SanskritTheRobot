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
    print('updating bot_data with saved user_info()')

    try:
        load_json_userdata(update, context)
        print('json read')
    except:
        text = "no json file found"

    user_id = update.effective_user.id

    if user_id not in context.bot_data:
        user = User()
        user.first(update.effective_user)
        context.bot_data.setdefault(user_id,  user )

        bot_user_data = context.bot_data.copy()
        try:
            bot_user_data.pop('openai_settings')
        except:
            pass
        write_json_data(bot_user_data, userdata_file_path())
        print('json write')
    return



#------------------------------------------------------------
#------------------------------------------------------------
api_stats_leaderboard_command   = CommandHandler("api_stats_leaderboard", all_users_api_stats_callback, run_async=True)
api_stats_command               = CommandHandler("api_stats", api_stats_callback, run_async=True)

#------------------------------------------------------------
#------------------------------------------------------------
