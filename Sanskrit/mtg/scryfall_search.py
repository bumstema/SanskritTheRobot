# Include Debug and System Tools
import traceback
import sys, os
import os.path
import io
import asyncio
import aiohttp
import json
import requests
from pathlib import Path
from requests import get
from json import loads
from shutil import copyfileobj
from io import BytesIO
from uuid import uuid4
from urllib.request import urlopen
from datetime import datetime, timedelta, time, date
from dataclasses import dataclass, field
from PIL import Image, ImageFilter, ImageOps, ImageChops
from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup

from telegram import Update, InputMediaPhoto, BotCommand
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import InlineQueryResultPhoto, InputMediaPhoto, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, CallbackContext, Filters
from telegram.ext import InlineQueryHandler, CallbackQueryHandler, MessageHandler

from ..ai.craiyon_img import append_images

#------------------------------------------------------------
def convert_matplot_to_pil_img( fig ):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300)
    buf.seek(0)
    return Image.open(buf)
    
#------------------------------------------------------------
def convert_pil_img_to_sendable_bio(pil_image: Image):
    bio = BytesIO()
    bio.name = f'temp.png'
    pil_image.save(bio, 'PNG')
    bio.seek(0)
    return bio

# ====================================================================
# ====================================================================
class Scryfall_Card:
    def __init__(self, card_data):
        self.card_data = card_data
        self.photo_url  = None
        self.thumb_url = None

        # Formatting Constants
        self.legalities = [ ("not_legal",'✖️'),("banned",'🚫'),("restricted",'⚠'),("legal",'✓')]
        self.listed_formats = ["standard", "modern", "pauper", "commander", "legacy", "vintage"]
        self.card_params = ['name', 'mana_cost', 'oracle_text', 'power', 'toughness', 'edhrec_rank', 'type_line']
        self.colour_pips = [('{R}','🔴'), ('{G}','🟢'), ('{U}','🔵'), ('{W}','⚪️'), ('{B}','⚫️'), ('{C}','⟡'),('{T}', ' ↷'),('{X}', '🅧')]
        self.phi_pips = [('{R/P}','(ϕ/🔴)'), ('{G/P}','(ϕ/🟢)'), ('{U/P}','(ϕ/🔵)'), ('{W/P}','(ϕ/⚪️)'), ('{B/P}','(ϕ/⚫️)')]
        self.mana_values = zip(range(11),['0️⃣','1️⃣','2️⃣','3️⃣','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣','🔟'])
        self.card_types = [('legendary','⭐️'),  ('artifact','🏺'), ('enchantment','📜'), ('creature', '🐕'), ('instant','💫'), ('sorcery','🪄'), ('adventure','📖'), ('land','🏝️'), ('battle','⚔️'), ('planeswalker','🚶')]
        
        self.im = self.img()
        self.card_caption = self.caption()
        
        #data["flavor_text"]
        #["image_uris"]["small"]["normal"]["png"]
        #["set_name"]
        #["edhrec_rank"]
        #["prices"]["usd"]["eur"]
        #["cmc"]
        #["mana_cost"]
        #["type_line"]
        #["oracle_text"]
        #["keywords"]
        #["rarity"]
        #["layout"] = "normal"  (seedcore)
        #["layout"] = "transform" (elesh norn)
        #["card_faces"]


    # generates and saves the images
    # -------------------------------------------------------------------------
    def img(self):
        acceptable_layouts = ['normal', 'adventure', 'saga']
        try:
            if (self.card_data['layout'] in acceptable_layouts):
                self.thumb_url = self.card_data['image_uris']['small']
                img_url = self.card_data['image_uris']['png']
                self.photo_url = img_url
                self.im = Image.open(requests.get(img_url, stream=True).raw)
            
            elif 'card_faces' in self.card_data:
                im = []
                for face_ in self.card_data['card_faces']:
                    self.thumb_url = face_['image_uris']['small']
                    img_url = face_['image_uris']['png']
                    self.photo_url = img_url
                    im.append(Image.open(requests.get(img_url, stream=True).raw))
                self.im = append_images(im, direction='horizontal')
            
            return self.im
            
            
        except:
            print(f'Something went wrong. \n')
            print(json.dumps(self.card_data, indent = 4))

    # -------------------------------------------------------------------------
    def caption(self):
    
        if 'card_faces' in self.card_data:
            self.card_caption = '\n\n'.join([ self.caption_per_face(card_face_) for card_face_ in self.card_data['card_faces']])
        else:
            self.card_caption = self.caption_per_face(self.card_data)
        
        # Rarity & Prices
        self.card_caption  += f"\n\n" + f"［{self.card_data['rarity'].title()} "
        self.card_caption  += f": ${self.card_data['prices']['usd']} · €{self.card_data['prices']['eur']}］"

        presorted_format = [(format_, self.card_data["legalities"][format_]) for format_ in self.listed_formats]
        
        card_legalities = [ f'{format_}'.title() + f'{legality_}' for format_, legality_ in presorted_format ]
        

        card_status  = f"\n"
        for (a_, b_, c_) in zip(card_legalities[::3], card_legalities[1::3], card_legalities[2::3]):
            card_status  +=  f"［" + a_ + f"· " + b_ + f"· " + c_+ f"］\n"
        
        for status, pip in self.legalities:
            card_status = card_status.replace(f'{status}', f'{pip}')

        self.card_caption += card_status

        return self.card_caption
        
    # ------------------------------------
    def caption_per_face(self, card_face):
    
        _ = [card_face.setdefault(param, f'') for param in self.card_params]
        
        for colour, pip in self.colour_pips:
            card_face['mana_cost'] = card_face['mana_cost'].replace(f'{colour}',f'{pip}')
            card_face['oracle_text'] = card_face['oracle_text'].replace(f'{colour}',f'{pip}')
            
        for phi, pip in self.phi_pips:
            card_face['mana_cost'] = card_face['mana_cost'].replace(f'{phi}',f'{pip}')
            card_face['oracle_text'] = card_face['oracle_text'].replace(f'{phi}',f'{pip}')
        
        for num, emoji in self.mana_values:
            card_face['mana_cost'] = card_face['mana_cost'].replace('{'+f'{num}'+'}',f'{emoji}')
            card_face['oracle_text'] = card_face['oracle_text'].replace('{'+f'{num}'+'}',f'{emoji}')
        
        
        type_emoji = ''
        type_on_card = card_face['type_line'].casefold()
        if f'—' in type_on_card:
            type_on_card = type_on_card.partition(f'—')[0]
        
        for type, emoji in self.card_types:
            if type in type_on_card:
                type_emoji += emoji

        
        if 'creature'.casefold() in card_face['type_line'].casefold():
            card_caption = f"{card_face['name']}: \t\t\t\t  {card_face['mana_cost']} \
                \n{type_emoji} {card_face['type_line']} \
                \n {card_face['oracle_text']} \
                \t ［{card_face['power']}/{card_face['toughness']}］"

        if 'land' in card_face['type_line'].casefold():
            card_caption = f"{card_face['name']}: \
                \n{type_emoji} {card_face['type_line']} \
                \n {card_face['oracle_text']}"

        if ('creature'.casefold() not in card_face['type_line'].casefold()) & ('land'.casefold() not in card_face['type_line'].casefold()):
            card_caption = f"{card_face['name']}: {card_face['mana_cost']} \
            \n{type_emoji} {card_face['type_line']}\
            \n {card_face['oracle_text']}"

        return card_caption
        
#-------------------------------------- scryfall get png
async def search_scryfall(search_query, inline=False):
    # In this example, we're looking for "Vindicate"
    #search_query = search_query
    # Load the card data from Scryfall
    card_results = loads(get(f"https://api.scryfall.com/cards/search?q={search_query}").text)
     
    # Get the image URL
    resulting_card = None
    possible_card_names = []
    inline_search_cards=[]
    
    if len(card_results['data']) >= 25:
        await asyncio.sleep(0.2)
        return (resulting_card, possible_card_names)
    
    if len(card_results['data']) <= 25:
        for card in card_results['data']:
            inline_search_cards.append(Scryfall_Card(card))
        if inline: return inline_search_cards

    for card in card_results['data']:
        possible_card_names.append(Scryfall_Card(card))
        formatted_name = card['name'].casefold().replace(',','').replace("'",'')
        formatted_query = search_query.casefold().replace(',','').replace("'",'')
        if (formatted_query in formatted_name) & (abs(len(formatted_name)-len(formatted_query)) < 2):
            resulting_card = Scryfall_Card(card)

    print(f"{possible_card_names = }")
    print(f"Ok!")
    await asyncio.sleep(0.2)
    return (resulting_card, possible_card_names)

# -----------------------------------------
def scryfall_img_callback(update, context):
    print(f'Scryfall Image Search || {update.effective_chat}')
    print(context.args)
    user_prompt = ' '.join(context.args)
    search_query = user_prompt
    search_and_select(search_query, update, context)

#------------------------------------------------------------
#------------------------------------------------------------
scryfall_img_command  = CommandHandler('scryfall', scryfall_img_callback)
scryfall_img_setbotcommands  = [ BotCommand( 'scryfall','Get image of MTG card from name.')]
#------------------------------------------------------------

# -------------------------------------------------
def search_and_select(card_query, update, context):
    chat_id = update.effective_chat.id

    (card, possible_card_names)  =  asyncio.run(search_scryfall(card_query))
    
    if card is None:

        context.bot.send_photo(chat_id, f"https://avatars.githubusercontent.com/u/22605579?s=200&v=4", caption= f'Multiple Cards Found. Please choose one:\n', reply_to_message_id=False, reply_markup=card_board_keyboard(possible_card_names) )
        
        return 'selector'

    send_scryfall_(card, update, context)

# ---------------------------------------
def send_scryfall_(card, update, context):
    try:
        update.message.reply_photo(convert_pil_img_to_sendable_bio(card.img()), caption = f'{card.caption()}')
        #  parse_mode='MarkdownV2'
        print(f'...Sent!')
        return
    except:
        context.bot.send_photo(f'{update.inline_query.update_id}', convert_pil_img_to_sendable_bio(card.img()), caption = f'{card.caption()}')
        return

#------------------------------------------------------------
def card_selector(update: Update, context: CallbackContext) -> None:
    print('card_selector()')

    query = update.callback_query
    user_id = update.effective_user.id
    print(f'{query.data = }')
    
    chat_id =  update.callback_query.message.chat.id
    msg_id = update.callback_query.message.message_id
    inline_msg_id = update.callback_query.inline_message_id
    
    search_name = query.data.replace('selector_','')
    (card, possible_card_names)  =  asyncio.run(search_scryfall(search_name))

    #send_scryfall_(card, update, context)
    #query.edit_message_text( text=card.card_caption )
    query.edit_message_media( InputMediaPhoto( card.photo_url, caption=card.card_caption))

    #context.bot.edit_message_media(chat_id=chat_id, message_id=msg_id, media=InputMediaPhoto( card.photo_url, caption=card.card_caption), reply_markup=None)
    #context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=msg_id, reply_markup=ReplyKeyboardRemove())


    #print(json.dumps(str(query), indent=4))
    #query.edit_message_media( card.photo_url )

    query.answer()
    #context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=msg_id, reply_markup=ReplyKeyboardRemove())
    
    return

#------------------------------------------------------------
#------------------------------------------------------------
card_selector_button = CallbackQueryHandler(card_selector, pattern = 'selector')
#------------------------------------------------------------


#------------------------------------------------------------
# Inline Query
#------------------------------------------------------------
def card_board(possible_cards, update, context) -> None:
    gboard = []
    for idx_, card in enumerate(possible_cards):
        gboard.append(
                    InlineQueryResultPhoto(
                        id=uuid4(),
                        title=f"{card.card_data['name']}",
                        caption = f"{card.card_caption}",
                        description=f"{card.card_caption}"[0:70]+"..." ,
                        thumb_url=f"{card.thumb_url}",
                        photo_url=f"{card.photo_url}"
                        )
                    )
    return gboard

#------------------------------------------------------------
def card_stack(possible_cards, update, context) -> None:
    gboard = []
    for idx_, card in enumerate(possible_cards):
        gboard.append(
                    InlineQueryResultArticle(
                        id=uuid4(),
                        title=f"{card.card_data['name']}",
                        #caption = f"{card.card_caption}",
                        description=f"{card.card_caption}"[0:70]+"..." ,
                        thumb_url=f"{card.thumb_url}",
                        photo_url=f"{card.photo_url}",
                        input_message_content = InputTextMessageContent(f"{card.card_caption}"+"\n"+f"{card.photo_url}")
                        )
                    )
    return gboard

#------------------------------------------------------------
def card_board_keyboard(possible_card_names):
    keyboard_buttons  = []
    keyboard_buttons = [ print(card_.card_data['name']) for card_ in possible_card_names]
    keyboard_buttons = [ [InlineKeyboardButton(card_.card_data['name'], callback_data='selector_'+card_.card_data['name'])] for card_ in possible_card_names]
    keyboard_markup = InlineKeyboardMarkup(keyboard_buttons)
    return keyboard_markup

# - ---------------------------------------
def choiceBoard() -> None:
    gboard = []
    how_to_scryfall = f'Example Inline Search.\n'
    how_to_scryfall += f'To Post a Card from Scryfall in Chat: \n\n'
    how_to_scryfall += '@SanskritTheRobot <card name>\n'
    how_to_scryfall += '(Result Bar will update automatically after a moment.)\n\n'
    how_to_scryfall += 'Example Command Search. \n'
    how_to_scryfall += 'To Search Scryfall in bot DMs:\n\n'
    how_to_scryfall += '/scryfall <card name>\n'
    how_to_scryfall += '(Select Card from List if Multiple Choices are returned.)\n\n'
    
    for idx_, choice_ in enumerate(['scryfall']):
        gboard.append(
                    InlineQueryResultArticle(
                        id=uuid4(),
                        title=f"Default Search Scryfall ... 🔍",
                        description=f"Type: <card name>"[0:70]+" ..." ,
                        thumb_url=f"https://avatars.githubusercontent.com/u/22605579?s=200&v=4",
                        input_message_content = InputTextMessageContent( how_to_scryfall ),
                        parse_mode='MarkdownV2'
                        )
                    )
    return gboard
    
#------------------------------------------------------------
def scryfall_lookup(update: Update, context: CallbackContext) -> None:
    print(f'Scryfall Inline Search || {update.effective_chat}' +
            f'Username: '+ str(update.effective_user.username) +
            f', ID: ' + str(update.effective_user.id)
        )

    query = update.inline_query.query
    card_query = str(query).casefold()
    print(f'Incoming Query: ' + str(query))
    results = None

    if  f'' == card_query:
        results = choiceBoard()

    else:
        print(f'{card_query = }')
        card_list  = asyncio.run(search_scryfall(card_query, inline=True))
        results = card_board(card_list, update, context)
        #results = card_stack(card_list, update, context)

    msg = update.inline_query.answer(results)
    print('.... InlineQueryAnswer Finished!')

#------------------------------------------------------------
#------------------------------------------------------------
scryfall_inlinequery = InlineQueryHandler(scryfall_lookup)
#------------------------------------------------------------

def passive_card_extractor(update, context):
    user_msg_text = update.message.text
    n_card_start = user_msg_text.count('[[[')
    n_card_end = user_msg_text.count(']]]')
    n_cards = (n_card_start + n_card_end) // 2

    if (n_card_end == n_card_start) & (n_cards > 0):
        card_names = []
        msg_text = user_msg_text
        for c_ in range(n_cards):
            card_start_char = msg_text.find('[[[')
            card_end_char = msg_text.find(']]]')
            card_name = msg_text[card_start_char+3:card_end_char]
            msg_text = msg_text[card_end_char:]
            search_and_select(card_name, update, context)

#------------------------------------------------------------
#------------------------------------------------------------
passive_card_extractor_handler = MessageHandler(Filters.text, passive_card_extractor)
#------------------------------------------------------------
#------------------------------------------------------------
