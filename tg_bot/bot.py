import logging
import os
import numpy as np
from aiogram import Bot, Dispatcher, executor, types
import imageio
from services import predict
import PIL
from static.config import BOT_TOKEN


API_TOKEN = BOT_TOKEN

# configure loggining
logging.basicConfig(level=logging.INFO)

# inizializate bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['help'])
async def send_welcome(message: types.Message):
	'''
	This handler will be called when user sends '/start' or '/help' command
	'''
	await message.reply(f'Hi {message.from_user.first_name}!\nI can paint female portraits!!\nJust send me your selfie!')

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
	await message.reply('Send me photo')

@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
	stickers_list = os.listdir(path='./static/welcome_stickers')
	choice = np.random.randint(len(stickers_list))
	sticker_to_send = stickers_list[choice]
	sti = open('./static/welcome_stickers/'+sticker_to_send, 'rb')
	await bot.send_sticker(message.chat.id, sti)
	await bot.send_message(message.chat.id, f'I mean you`re silly, right {message.from_user.first_name}?\nI understand you want to chat, but that`s not what I`m made for.\nJust send me photo!')


@dp.message_handler(content_types=['photo'])
async def reply_photo(message: types.Message):
	file_name = './images/' + str(message.from_user.id) + 'upload.jpg'
	image_path = await message.photo[-1].download(file_name)
	predict(file_name)
	caption = 'Here you go! Your photo has been processed!'
	with open(file_name, 'rb') as f:
		await bot.send_photo(chat_id=message.chat.id, photo=f, caption=caption)





if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
