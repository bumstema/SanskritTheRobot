import telegram
from telegram import Update, BotCommand, InputMediaPhoto
from telegram.ext import Updater, CallbackContext, Filters, MessageHandler,  ConversationHandler, CommandHandler, CallbackQueryHandler
from telegram import ReplyKeyboardRemove, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup



import os
from pathlib import Path
import random
from random import randint
import time

import cv2
import torch
import numpy as np
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from PIL import Image, ImageFilter, ImageOps, ImageChops
from PIL import GifImagePlugin, ImageSequence

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionControlNetPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, ControlNetModel, AutoencoderKL
from diffusers import  DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,  LMSDiscreteScheduler, DDPMScheduler, DEISMultistepScheduler
from diffusers import PNDMScheduler, DDIMScheduler, HeunDiscreteScheduler

from transformers import pipeline
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from controlnet_aux import PidiNetDetector, HEDdetector
from controlnet_aux.processor import Processor


from .api_stats 		import User, user_info, userdata_file_path
from .openai_img 		import get_photo_from_message


from ..utils.utils 		import read_json_data, write_json_data, unpack_nested_list
from ..utils.utils 		import convert_pil_img_to_sendable_bytes, get_bytes_from_a_incomming_photo
from ..utils.utils 		import cv_image_to_pil, pil_image_to_cv, progress

from ..const.ai_consts 	import PALETTE

from .craiyon_img       import append_images


##------------------------------------------------------------#------------------------------------------------------------
##------------------------------------------------------------#------------------------------------------------------------
    ##  Saved MODELS
    
    #"Fictiverse/Stable_Diffusion_PaperCut_Model"
    #"dallinmackay/Van-Gogh-diffusion"
    #"runwayml/stable-diffusion-v1-5"


    ## Possible Choices OF SCHEDULER
    
    #pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)



    #processor_id = 'shuffle'            # Wonky swirls that seem random
    #processor_id = 'softedge_hed'       # Thick lines, clear edges, internal structure appearing, mega blurry
    #processor_id = 'softedge_hedsafe'   # Thick lines, edge outlines, shadows appearing
    #processor_id = 'softedge_pidinet'   # Thick lines, connected at edges, some internal structure, blurry
    #processor_id = 'scribble_hed'       # Thick lines, artifacts, does not capture all edges
    #processor_id = 'scribble_pidinet'   # Thick lines with jumping artifacts
    #processor_id = 'lineart_coarse'     # Thin lines with sharp outline and faint inner detail
    #processor_id = 'lineart_realistic'  # Medium width lines, sharp contours of objects and lots of inner detail (slow)
    #processor_id = 'lineart_anime'      # Not lines - colours filled in with style of "Xerox copy" (fast)

    # 'depth_leres'
    # 'depth_leres++'
    # 'depth_midas'
    # 'depth_zoe'


##------------------------------------------------------------
def sd_file_path():
    return Path('/Users/python/python/stable-diffusion/pretrained_models/')


##------------------------------------------------------------
def file_path(update, context, file_type='.jpeg'):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if '.jpeg' in file_type:
        return f'./Sanskrit/data/ai_generations/stable_diffusion_{user_id}_{chat_id}_{context.bot_data[user_id].save_number()}.jpeg'
    if '.mp4' in file_type:
        return f'./Sanskrit/data/animations/mp4_{user_id}_{chat_id}.mp4'
    if '.gif' in file_type:
        return f'./Sanskrit/data/animations/gif_{user_id}_{chat_id}.gif'
    if '.test' in file_type:
        return f'./Sanskrit/data/animations/IMG_5349.jpg'
    if '.prefix' in file_type:
        return Path(f'./Sanskrit/data/animations/')


#------------------------------------------------------------
def select_pipe_scheduler( pipe ):

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    inference_steps = 20
    if random.random() >= 0.75:
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
        inference_steps = 40
        if random.random() <= 0.5:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            inference_steps = 40
    return (pipe.scheduler, inference_steps)


##------------------------------------------------------------#------------------------------------------------------------
##------------------------------------------------------------#------------------------------------------------------------
def stable_diffusion_callback(update, context):

    print(f'Stable Diffusion || {update.effective_chat}')
    user_info(update, context)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_seed = int(time.time())
    print(f'{user_seed}')

    print(context.args)
    user_prompt = ' '.join(context.args)
    print(user_prompt)
    if user_prompt == '':
        msg = update.message.reply_text(f"How to Generate Art: \n1. Begin with the command. \n2. Include tags or description. \n\n(Example)\n/aiart fox in space with the stars \n\n Ok, now you try! Each picture takes ~1min to generate.\n\nPlease only send (1) one prompt at a time! Sometimes messages fail to be received. Retry again if the 'sending screen' did not appear after 30sec.")
        user_prompt = f'A curious user clicking on robot commands.'
        return
        
    neg = f'absurd resolution'
    h,w = 512,512
    #h,w = 360,640
    
    #  For gay621  w, h = 832, 448 || w, h = 960, 640
    
    # -- Remove settings and write ONLY user data to Json File
    context.bot_data[user_id].add_stable_diffusion_call(user_prompt)
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())

    msg = context.bot.send_photo(f'{update.effective_chat.id}', open('./Sanskrit/ai/ai_photos/computer_vision_logo.png','rb'), caption = f'Sending prompt: \"{user_prompt}\" to Stable Diffusion Model. ⏳', timeout=360)
    
    sd_model = f'{sd_file_path()}' + f'runwayml/stable-diffusion-v1-5.safetensors'
    vae_model = f'{sd_file_path()}' + f'/fixYourColorsVAE_vaeFtMse840000Ema.pt'
    
    pipe = StableDiffusionPipeline.from_single_file(f'{sd_model}', safety_checker=None)
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.enable_attention_slicing()
    pipe.scheduler, inference_steps =  select_pipe_scheduler(pipe)
    
    
    #vae = AutoencoderKL.from_single_file(vae_model).to("mps")
    #pipeline.vae = vae
    
    pipe = pipe.to("mps")
    generator = torch.Generator(device="mps").manual_seed(user_seed)

    
    image = pipe(user_prompt,
        generator=generator,
        num_inference_steps=inference_steps,
        height=h,
        width=w,
        guidance_scale=6,
        negative_prompt=neg).images[0]
        
    image.save(file_path(update, context))

    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open('./Sanskrit/ai/ai_photos/computer_vision_logo-inverted.png','rb'), caption = f'Successful generation! Now sending to chat ... ⌛️'), timeout=360)
    
    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open(  file_path(update, context), 'rb'), caption = f'Prompt: \"{user_prompt}\"'), timeout=360)
    
    print(f'---Sent!')
    del msg, image
    return

#------------------------------------------------------------
stable_diffusion_command = CommandHandler("aiart", stable_diffusion_callback, run_async=True)
stable_diffusion_setbotcommands  = [ BotCommand( 'aiart','..prompt the Ai to generate an image.')]
#------------------------------------------------------------#------------------------------------------------------------
#------------------------------------------------------------#------------------------------------------------------------



#------------------------------------------------------------#------------------------------------------------------------
def pix2pix_callback(update, context):

    print(f'Pix2Pix || {update.effective_chat}')
    raw_prompt = update.message.caption
    print(f'{raw_prompt = }')
    
    select_controlnet = False
    select_pix2pix = False
    if f'controlnet' in raw_prompt:
        select_controlnet = True
    if f'pix2pix' in raw_prompt:
        select_pix2pix = True
    
    user_prompt = raw_prompt.replace('/pix2pix', '').replace('pix2pix', '').replace('\n ', '').strip()
    user_prompt = user_prompt.replace(f'controlnet', f'').strip()
    if user_prompt == "": user_prompt =  f"turn it into a fox"
    print(f'{user_prompt = }')

    user_info(update, context)
    user_id = update.effective_user.id
    context.bot_data[user_id].add_stable_diffusion_call(user_prompt)
    
    # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())

    # First Message - Open Link to Telegram
    msg = context.bot.send_photo(f'{update.effective_chat.id}', open('./Sanskrit/ai/ai_photos/computer_vision_logo.png','rb'), caption = f'Using prompt: \"{user_prompt}\". ⏳' , reply_to_message_id = update.message.message_id, timeout=360)

    sent_photo = get_photo_from_message(update, context)
    with Image.open(sent_photo) as sent_image:
        if select_controlnet:
            diffused_img = controlnet_image_modifier(user_prompt, [sent_image])[0]
        else:
            diffused_img = instruct_pix2pix(user_prompt, sent_image)
        
    diffused_img.save(file_path(update, context))

    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( open('./Sanskrit/ai/ai_photos/computer_vision_logo-inverted.png','rb'), caption = f'Successful generation! Now sending to chat ... ⌛️'), timeout=360 )

    context.bot.edit_message_media( chat_id = f'{update.effective_chat.id}', message_id = msg.message_id, media = InputMediaPhoto( convert_pil_img_to_sendable_bytes(diffused_img), caption = f'Prompt: \"{user_prompt}\"'), timeout=360 )
    
    print(f'---Sent!')
    return
    
#------------------------------------------------------------
pix2pix_command = MessageHandler(Filters.photo & (Filters.caption_regex(r'pix2pix') | Filters.caption_regex(r'controlnet'))  , pix2pix_callback, run_async=True)
#------------------------------------------------------------#------------------------------------------------------------
#------------------------------------------------------------#------------------------------------------------------------



#------------------------------------------------------------
def instruct_pix2pix(prompt: str, img : Image, neg='') -> Image:
    print(f'-- Instruct pix2pix || ')
    model_id = "timbrooks/instruct-pix2pix"

    if prompt == "": prompt = "turn it into a fox"
    print(f'pix2pix prompt: {prompt}')
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
    pipe.to("mps")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.enable_attention_slicing()
    
    pipe.scheduler, inference_steps =  select_pipe_scheduler( pipe )

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = ImageOps.contain(img, (512,512))
    
    image = pipe(prompt, image=img, num_inference_steps=inference_steps, guidance_scale=6, negative_prompt=neg).images[0]

    return image

#------------------------------------------------------------
def controlnet_image_modifier(prompt, img_frames, neg=f'') :
    print(f'-- Controlnet || ')
    pretrained_model = f'{sd_file_path()}' + f'/runwayml/stable-diffusion-v1-5.safetensors'
    from_pretrained = False
    preprompt = f''
    postprompt = f', high resolution, best quality'
    
    neg = f'absurd resolution'
    
    if '#vangogh' in prompt:
        pretrained_model = "dallinmackay/Van-Gogh-diffusion"
        preprompt = f'lvngvncnt, '
        from_pretrained = True
        
    elif '#papercut' in prompt:
        pretrained_model = "Fictiverse/Stable_Diffusion_PaperCut_Model"
        preprompt = f'PaperCut '
        from_pretrained = True
        controlnet_model = "lllyasviel/sd-controlnet-normal"
        gif_frames  = preprocess_normal_estimation(img_frames)
        
    else:
        #controlnet_model = "lllyasviel/sd-controlnet-depth"
        #gif_frames  = preprocess_depth_estimation(img_frames)
        
        #controlnet_model = "lllyasviel/sd-controlnet-hed"
        #gif_frames  = preprocess_edge_estimation(img_frames)
        
        #controlnet_model = "lllyasviel/sd-controlnet-normal"
        #gif_frames  = preprocess_normal_estimation(img_frames)
    
        controlnet_model = "lllyasviel/control_v11p_sd15_softedge"
        gif_frames  = preprocess_edge_estimation(img_frames)


    n_sqr = 9
    n_box_panels = len(img_frames)//n_sqr

    
    ctrlnet_imgs = [ ImageOps.contain(pic, (512,512)) for pic in gif_frames ]
    ctrlnet_imgs = [ ImageOps.scale(pic, 2/3 ) for pic in ctrlnet_imgs ]
    
    print(f'{np.array(ctrlnet_imgs[0]).shape = }')
    [h, w, c] = np.array(ctrlnet_imgs[0]).shape
    
    ctrlnet_box_imgs = [ n_by_n_image_reduction(ctrlnet_imgs[(i*n_sqr):((i+1)*n_sqr)]) for i in range(0, n_box_panels) ]

    print(f'{np.array(ctrlnet_box_imgs[0]).shape = }')

    if prompt == "": prompt = "cartoonize this image"
    prompt = " ".join([word for word in prompt.split() if '#' not in word])
    prompt = prompt.replace(f'controlnet', f'')
    prompt = preprompt + f'{prompt}' + postprompt
    
    print(f'pix2pix prompt: {prompt}')
    print(f'pretrained: {pretrained_model}')
    print(f'controlnet: {controlnet_model}')
    
    controlnet = ControlNetModel.from_pretrained(controlnet_model)
    controlnet.to("mps")
    
    if not from_pretrained:
        pipe = StableDiffusionControlNetPipeline.from_single_file(pretrained_model, controlnet=controlnet)
    if from_pretrained:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(pretrained_model, controlnet=controlnet)

    pipe.to("mps")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.enable_attention_slicing()
    
    pipe.scheduler, inference_steps =  select_pipe_scheduler(pipe)
    
    sd_block_frames = [ pipe(prompt, image=pic, num_inference_steps=inference_steps, guidance_scale=6, negative_prompt=neg).images[0] for pic in ctrlnet_box_imgs ]

    sd_block_frames = [ np.array(pic) for pic in sd_block_frames ]
    sd_frames = [ split_image(pic, h, w) for pic in sd_block_frames ]
    sd_frames = [ Image.fromarray(np.array(pic)) for pic in unpack_nested_list(sd_frames) ]
    #sd_frames = [ ImageOps.scale(pic, 2.0) for pic in sd_frames ]

    return sd_frames
    
    
#------------------------------------------------------------
def split_image(im, M, N):
    return [im[x:x+M,y:y+N,:] for x in range(0,im.shape[0],M)  for y in range(0,im.shape[1],N)]


##------------------------------------------------------------
def n_by_n_image_reduction(n_by_n_images):
    n = int(np.sqrt(len(n_by_n_images)))
    x_by_n = [append_images(n_by_n_images[i:(i+n)], direction='horizontal') for i in range(0,len(n_by_n_images), n)]
    final = append_images(x_by_n, direction='vertical')

    return final


#------------------------------------------------------------#------------------------------------------------------------
#------------------------------------------------------------
def preprocess_segmentation_estimation(gif_imgs):

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    start_time = datetime.now()
    print(f"Starting Segment Estimation...")
    img_shape = (512, 512)
    segment_gif = []
    
    for index, pic in enumerate(gif_imgs):
    
        image = ImageOps.contain(pic, img_shape).convert('RGB')
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
          outputs = image_segmentor(pixel_values)

        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(PALETTE):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)
        
        segment_gif.append(image)
        progress(index+1, len(gif_imgs))
        
    return segment_gif


#------------------------------------------------------------
def preprocess_normal_estimation(gif_imgs):

    depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

    start_time = datetime.now()
    print(f"Starting Normal Estimation...")

    img_shape = (512, 512)
    img_shape = gif_imgs[0].size
    normal_gif = []

    for index, pic in enumerate(gif_imgs):
        image = ImageOps.contain(pic, img_shape)
        image = depth_estimator(image)['predicted_depth'][0]

        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.35

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        image = ImageOps.contain(image, img_shape).convert("RGB")
        normal_gif.append(image)
        progress(index+1, len(gif_imgs))
        
    return normal_gif


#------------------------------------------------------------
def preprocess_depth_estimation(gif_imgs):

    depth_estimator = pipeline('depth-estimation', "vinvino02/glpn-nyu")

    start_time = datetime.now()
    print(f"Starting Depth Estimation...")
    
    img_shape = (512, 512)
    depth_gif = []
    
    for index, pic in enumerate(gif_imgs):
        image = ImageOps.contain(pic, img_shape)
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        #og_img = image
        #image = ImageChops.multiply(image, image)
        #image = ImageChops.multiply(image, image)
        #image = ImageChops.multiply(og_img, image)
        depth_gif.append(image)
        progress(index+1, len(gif_imgs))
        
    return depth_gif


#------------------------------------------------------------
def preprocess_edge_estimation(gif_imgs):

    start_time = datetime.now()
    print(f"Starting Edge Estimation...")
    
    img_shape = (512, 512)
    edge_gif = []

    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    
    for index, pic in enumerate(gif_imgs):
        img = ImageOps.contain(pic, img_shape).convert("RGB")
        image = processor(img, to_pil=True, safe=True)
        edge_gif.append(image)
        progress(index+1, len(gif_imgs))
        
    return edge_gif

    
#------------------------------------------------------------#------------------------------------------------------------
#------------------------------------------------------------#------------------------------------------------------------


ASK_FOR_PROMPT = 3

#------------------------------------------------------------
def grab_gif(update, context):
    print(f"Downloading gif from telegram...")
    
    file_id = update.effective_message.animation.file_id
    context.bot.get_file(file_id).download(Path(file_path(update, context, file_type='.mp4')))
    return
    
    
#------------------------------------------------------------
def load_mp4_as_frames(update, context):
    print(f"Loading video file...")

    vidcap = cv2.VideoCapture(str(Path(file_path(update, context, file_type='.mp4'))))

    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_frames = vidcap.get(cv2.CAP_PROP_FPS)
    ms_per_frame = (1/fps_frames)*1000
    print( f"Total Frames: {n_frames}  Frames/sec: {fps_frames}  ms/Frame: {ms_per_frame}")
    
    gif_frames = []

    success,image = vidcap.read()
    while success:
        gif_frames.append(cv_image_to_pil(image))
        success,image = vidcap.read()
    
    gif_frames = (gif_frames[:n_frames:2])
    ms_per_frame = ms_per_frame * 2
    
    return gif_frames, ms_per_frame
    

#------------------------------------------------------------
def save_gif_frames(update, context, gif_frames, ms_per_frame):
    print(f"Now saving gif...")
    gif_frames[0].save(Path(file_path(update, context, file_type='.gif')), format='GIF', append_images=gif_frames[1:], save_all=True, loop=0, duration= ms_per_frame, disposal=0, optimize=True)
    return
    

#------------------------------------------------------------
def send_gif(update, context):
    print(f"Sending saved gif to chat...")
    context.bot.send_animation(f'{update.effective_chat.id}', open(file_path(update, context, file_type='.gif'),'rb') , reply_to_message_id = update.message.message_id)
    print(f"Sent!")
    return


#------------------------------------------------------------
def gif_detected(update, context):
    print(f"Gif Detected || {update.effective_chat}")
    grab_gif(update, context)
    
    update.message.reply_text(f"\"Gif-Frame-Replacement\".\n\nThis mode uses stable diffusion to change the subject of each frame in the animation that was just sent.\n\nPlease send a prompt for the model (or \"cancel\" to stop).")
    
    return ASK_FOR_PROMPT
    
    
#------------------------------------------------------------
def gif_prompt_and_submit(update, context):
    print(f"Gif Prompt || {update.effective_chat}")
    
    user_info(update, context)
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_seed = np.abs(user_id + chat_id)

    print(f'{update.message.text}')
    user_prompt = f'{update.message.text}'
    print(user_prompt)

    if f'cancel' in user_prompt.casefold(): return
    
    if user_prompt == '': user_prompt = f'A curious user clicking on robot commands.'
    context.bot_data[user_id].add_stable_diffusion_call(user_prompt)

    # -- Remove settings and write ONLY user data to Json File
    bot_user_data = context.bot_data.copy()
    bot_user_data.pop('openai_settings')
    write_json_data(bot_user_data, userdata_file_path())
    
    gif_frames, ms_per_frame = load_mp4_as_frames(update, context)
    
    text = f"\"Gif-Frame-Replacement\".\n\nframes: {len(gif_frames)}\n\nprompt: {user_prompt}\n\nNow running model...⏳"
    context.bot.send_message(f'{update.effective_chat.id}', text, reply_to_message_id = update.message.message_id)
    
    sd_frames = controlnet_image_modifier(user_prompt, gif_frames)
    
    save_gif_frames(update, context, sd_frames, ms_per_frame)
    
    try:
        send_gif(update, context)
    except:
        print(f'Failed - broken pipe (retrying)')
        send_gif(update, context)
        
    return


#------------------------------------------------------------
def cancel_gif(update, context):
    return 

    
#------------------------------------------------------------#------------------------------------------------------------
resend_gif_command = MessageHandler( ( Filters.regex(r'resend gif') & Filters.chat_type.private ), send_gif, run_async=True)

#------------------------------------------------------------#------------------------------------------------------------
grabgif_command = MessageHandler( (((Filters.document.gif & ~ Filters.command(False))) & Filters.chat_type.private ), gif_detected, run_async=True)
#------------------------------------------------------------#------------------------------------------------------------
#------------------------------------------------------------
submit_gif_to_stablediffusion_convo = ConversationHandler(
    entry_points = [grabgif_command],
    states = {
        ASK_FOR_PROMPT : [
            MessageHandler(
                (Filters.text & Filters.chat_type.private),
                gif_prompt_and_submit)
            ]
        },
    fallbacks = [MessageHandler(Filters.all, cancel_gif)]
    )

