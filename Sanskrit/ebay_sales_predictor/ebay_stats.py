

# Include Debug and System Tools
import traceback
import sys, os
import os.path
from pathlib import Path
import io
from io import BytesIO

# Include JSON for importing search results from website
import json

# import urllib library
from urllib.request import urlopen

from datetime import datetime, timedelta, time, date
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import t
from math import trunc

# import processing library
from PIL import Image, ImageFilter, ImageOps, ImageChops

# Computer Vision Module - OpenCV
#import cv2 as cv

# Asynchronous execution library
import asyncio


from dataclasses import dataclass
from typing import List



import aiohttp
from bs4 import BeautifulSoup


# Library for Telegram classes
from telegram import Update, InputMediaPhoto, BotCommand
from telegram.ext import Updater, CommandHandler, CallbackContext




# ------- ChatGPT Code --------
@dataclass
class EbayListing:
    name: str
    price: float
    url: str
    listed_date: datetime
    searched_query:  List[ str ]


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



# ------- ChatGPT Code --------
def find_outliers_in_data(date_list):

    days_extract = lambda x : x.days
    date_list = [ days_extract(datetime.now()+timedelta(days=1) - x) for x in date_list]
    print(f'{date_list}')
    # calculate mean and standard deviation using NumPy
    mean = np.mean(date_list)
    std_dev = np.std(date_list, ddof=1)

    print(f'{mean}, {std_dev}, {len(date_list)}')
    # calculate Grubbs' test statistic and critical value
    alpha = 0.05 # significance level
    t_value = t.ppf(1 - alpha/(2*len(date_list)), len(date_list) - 2)
    G = [(date - mean)/std_dev for date in date_list]
    critical_value = ((len(date_list) - 1)/(len(date_list)**0.5)) * ((t_value)/(len(date_list) - 2 + t_value**2)**0.5)

    # identify all the outlier dates
    outlier_indices = []
    for i in range(len(G)):
        if abs(G[i]) > critical_value:
            outlier_indices.append(i)
    print(f'{outlier_indices}')
    if len(outlier_indices) == 0:
        print("No date is an outlier.")
        return []
    else:
        #print(f"The dates {[date_list[i].strftime('%Y-%m-%d') for i in outlier_indices]} are outliers.")
        return outlier_indices


# ------- ChatGPT Code --------

def draw_scatter_plot(data: List[EbayListing], searched_term : str):



    # Convert the list of EbayListing objects to a Pandas DataFrame
    df = pd.DataFrame([vars(x) for x in data])



    # Calculate the mean of each column
    means = df['price'].mean()
    stds = df['price'].std()

    stats = {'mean_price': means , 'price_std_dev': stds}

    # Perform FFT on date series to find most probable sale frequency
    # Convert the dates to a Pandas Series
    s = df['listed_date']
    # Count the number of occurrences of each date
    counts = s.value_counts()
    # Create a new DataFrame with the date counts
    df_2 = pd.DataFrame({'count': counts})
    # Resample the DataFrame to a regular grid with a frequency of 1 day
    df_resampled = df_2.resample('1D').sum()
    # Perform FFT
    Y = np.fft.fft(df_resampled['count'])
    freq = np.fft.fftfreq(len(df_resampled), 1)
    # Find frequency where max FFT value occurs
    max_idx = np.argmax(np.abs(Y))
    print(f"{max_idx}, {np.abs(Y)}")
    # Remove the most probable 'zero day' sale
    Y[max_idx] = 0.0
    # Resolve fft peak
    max_idx =  np.argmax(np.abs(Y))
    # Get the magnitude of the FFT value at the maximum index
    # Find all peaks in the FFT
    #print(f"{max_idx}, {np.abs(Y)}")
    #peaks, info = find_peaks(np.abs(Y))
    #max_idx = np.argmax(np.abs(Y[peaks]))
    #print(f"{max_idx}, {np.abs(Y[peaks])}")

    print(f"{max_idx}, {np.abs(Y)}")
    max_freq = freq[max_idx]



    date_range = df['listed_date'].max() - df['listed_date'].min()
    average_days_per_sale = (date_range.days / len(df['listed_date']))

    inverted_sales_frequency = (1/max_freq)
    sale_delta = timedelta(days=average_days_per_sale)

    stats.update( {'average_days_per_sale': average_days_per_sale , 'fft_days_per_sale': inverted_sales_frequency})

    print(f"{means}, {stds}, {average_days_per_sale}, {inverted_sales_frequency} ")

    # Create a scatter plot of price vs listed_date
    fig, ax = plt.subplots(constrained_layout=False)
    ax.scatter(df['listed_date'], df['price'], alpha=0.5, c='b', s=20.5)

    # Set the title and labels for the plot
    plt.title(f"({len(df['price'])}) Sold Items for \"{searched_term}\" on eBay.ca", fontsize=14)
    plt.xlabel("Sale Date of Listing", fontsize=15)
    plt.ylabel("Price ($CAD)", fontsize=15)


    # Format the x-axis tick labels as dates
    date_format = mdates.DateFormatter('%Y-%m-%d')


    # Rotate the x-axis labels
    plt.xticks(rotation=-45)

    # Add gridlines
    plt.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=1.0, alpha=0.5)
    # Set x-axis formatter and locator for every 7 days
    months = mdates.MonthLocator(bymonthday=1,interval=1)
    month_formatter = mdates.ConciseDateFormatter(months)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(month_formatter)

    weekdays = mdates.WeekdayLocator(interval=7)
    days = mdates.WeekdayLocator(interval=1)
    week_formatter = mdates.ConciseDateFormatter(weekdays)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(week_formatter)
    plt.grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5)




    # Plot a horizontal line at the mean value of column Price
    ax.axhline(y=means, color='r', linestyle='-', linewidth=1.0)
    ax.text(df['listed_date'].max(), means, f"x̅= ${means:.2f}\n", va='center', ha='right', color='r', fontsize=10)

    # Plot a horizontal line at the standard deviation value of mean Price
    ax.axhline(y=means+stds, color='r', linestyle='--', alpha=0.75, linewidth=0.75)
    ax.axhline(y=means-stds, color='r', linestyle='--', alpha=0.75, linewidth=0.75)
    ax.text(df['listed_date'].max(), means+stds, f"\nσ= ±${stds:.2f}", va='center', ha='right', color='r', fontsize=9)
    ax.text(df['listed_date'].max(), means-stds, f"σ= ±${stds:.2f}\n", va='center', ha='right', color='r', fontsize=9)



    # Plot vertical line with days per sale info
    ax.axvline(x=df['listed_date'].min(), color='b', linestyle='--')
    ax.text(df['listed_date'].min() + sale_delta, df['price'].max() , f"Δ days/sale: {average_days_per_sale :.2f}", va='center', ha='left', color='b', fontsize=9)
    ax.text(df['listed_date'].min() + sale_delta, df['price'].max() - (stds/2) , f"FFT days/sale: {inverted_sales_frequency:.2f}", va='center', ha='left', color='b', fontsize=9)

    plt.tight_layout()
    #fig = plt

    #pil_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb() )
    # Display the plot
    #plt.show()
    #sleep(5)

    print(f"Starting conversion to pil image ")
    pil_img = convert_matplot_to_pil_img(fig)

    #await asyncio.sleep(0.1)
    print(f"Plot to Pil complete.")
    data_sent_back = {'img': pil_img,'stats': stats}
    #await asyncio.sleep(0.1)
    return data_sent_back


    # Allow time for plot to be displayed
    #await asyncio.sleep(0.1)


# ------- ChatGPT Code --------

async def fetch(session, url, params):
    async with session.get(url, params=params) as response:
        return await response.text()

#================================================================
async def search_ebay_ca(search_query):

    search_words = search_query.casefold().split(" ")
    print(f"Searching ebay.ca for {search_words}... \n")

    base_url 	= f"https://www.ebay.ca/sch/i.html"
    params = {"_nkw": search_query, "_pgn": 1, "_ipg": 100, "_skc": 50, "LH_BIN": 1, "sacat": 0, "LH_Sold": 1}

    async with aiohttp.ClientSession() as session:
        html = await fetch(session, base_url, params)

    soup = BeautifulSoup(html, "html.parser")
    soup_ebay_listings = soup.find_all("li", {"class": "s-item"})
    print(f'Total listings found: {len(soup_ebay_listings)}')

    item_listings = []
    for listing in soup_ebay_listings:
        title 	= listing.find("div", {"class": "s-item__title"}).text.strip()


        unfound_terms = [ term for term in search_words if term not in title.casefold() ]
        print(f'{title[0:50]} \t {unfound_terms} \t {not unfound_terms}')
        if unfound_terms:
            #print(f'{title} \t {unfound_terms}')
            continue

        price 	= listing.find("span", {"class": "s-item__price"}).text.strip().replace(",", "").replace("C", "").replace("$", "")
        if "to" in price:
            continue
        else:
            price = float(price)
        link = listing.find("a", {"class": "s-item__link"})["href"]
        # Skip over ads
        if "ebay.com" in link:	continue

        sell_date = listing.find('span', {'class': 'POSITIVE'}).text.strip().replace("Sold  ", "")
        date_sold = datetime.strptime(sell_date, "%d %b %Y")
        if datetime.now() - date_sold > timedelta(days=90): continue


        # Create a new EbayListing instance for each listing and append it to the list
        ebay_listing = EbayListing(name=title, price=price, url=link, listed_date=date_sold, searched_query=search_words)
        item_listings.append(ebay_listing)

    item_listings.sort(key = lambda x:  datetime.now()-x.listed_date )

    outlier_indicies = 1
    while outlier_indicies:
        date_list = [x.listed_date for x in item_listings]
        outlier_indicies = find_outliers_in_data(date_list)
        for index in outlier_indicies[::-1]: item_listings.pop(index)

    [print(f'{item.price} \t {item.listed_date.isoformat()} \t {item.name[0:50]} ') for item in item_listings]
    await asyncio.sleep(0.1)
    return (item_listings)



"""
# ------- ChatGPT Code --------
async def main():
    search_query = "ty beanie boos frost fox"
    # Search Ebay for Data
    ebay_listings = await search_ebay_ca(search_query)
    # Draw a scatter plot of price vs listed_date
    await draw_scatter_plot(ebay_listings, search_query)
"""

def ebay_sales_stats_callback(update, context):
    print(context.args)
    user_prompt = ' '.join(context.args)
    search_query = user_prompt

    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(main())

    # Search Ebay for Data
    #ebay_listings = await search_ebay_ca(search_query)
    # Draw a scatter plot of price vs listed_date
    #await draw_scatter_plot(ebay_listings, search_query)


    try:
        ebay_listings  =  asyncio.run(search_ebay_ca(search_query))
        returning_data =    draw_scatter_plot(ebay_listings, search_query)

        print(f"Plot and stats ready to be sent.")
        pil_img = returning_data['img']
        stats = returning_data['stats']

        text = f'📊 eBay․ca Sales Statistics for:\n\n\"{user_prompt}\"\n\n💵 Sale Price: ${stats["mean_price"]:.2f} ± ${stats["price_std_dev"]:.2f} (CAD)'
        text += f'\n📅 Days per Sale: {stats["average_days_per_sale"]:.2f}.\n🔁 FFT Inverse Sales Frequency: {stats["fft_days_per_sale"]:.2f}.'

        sendable_frame = convert_pil_img_to_sendable_bio(pil_img)
        context.bot.send_photo(f'{update.effective_chat.id}', photo=sendable_frame, caption = f'{text}')
        return
    except:
        update.message.reply_text(f'Something went wrong D:')


# ------- ChatGPT Code --------
"""
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
"""

#------------------------------------------------------------
#------------------------------------------------------------
ebay_sales_command  = CommandHandler('search_ebay', ebay_sales_stats_callback)
ebay_setbotcommands  = [ BotCommand( 'search_ebay','Get sales statistics for items.')]

#------------------------------------------------------------
#------------------------------------------------------------
