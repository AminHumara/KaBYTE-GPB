import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.types import FSInputFile

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

import torch
from transformers import (
    AutoImageProcessor, AutoModelForObjectDetection, pipeline
)

bot = Bot(token='6953190065:AAGN6-No7kafX77eIrcjPxUdmHESC5sddV0')
dp = Dispatcher()

label2text = {
    "neutral": 'бодрствует',
    "microsleep": 'спит',
    "yawning": 'зевает',
}

# Load processor and model
processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


async def send_message(chat_id, text):
    await bot.send_message(chat_id, text)


async def classify_person(image, chat_id):
    # Process inputs
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Post-process object detection results
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.6)[0]

    filtered_detections = [(score, label, box) for score, label, box in
                           zip(results["scores"], results["labels"], results["boxes"]) if
                           model.config.id2label[label.item()] == 'person' and score.item() > 0.88]

    if filtered_detections:
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display the image
        ax.imshow(image)

        # Store cropped images of persons
        cropped_images = []
        result_message = ""  # Initialize the result message

        # Iterate over the results, only for persons with high confidence
        for i, (score, label, box) in enumerate(filtered_detections):
            box = [round(coord, 2) for coord in box.tolist()]
            # Draw bounding box on the image
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

            # Annotate with numbers and background
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1)
            ax.text(box[0], box[1], str(i + 1), color='black', fontsize=12, ha='left', va='top', bbox=bbox_props)

            # Crop the person from the original image
            person_img = image.crop((box[0], box[1], box[2], box[3]))  # Corrected cropping coordinates
            cropped_images.append(person_img)

            # Classify the cropped image
            classifier = pipeline("image-classification", model=r'checkpoint-3110')
            prediction = classifier(person_img)[0]
            # Get the label and score
            label_text = label2text[prediction['label']]
            score = int(100 * round(prediction['score'], 2))
            # Add the classification result to the result message
            result_message += f"Человек на картинке {i + 1} с вероятностью {score}% {label_text}.\n"

        # Show the image with bounding boxes
        plt.axis('off')
        plt.savefig("output.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Send the result message
        await send_message(chat_id, result_message)
        return True
    else:
        await send_message(chat_id, "Людей на фото не обнаружено.")
        return False


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer('''Привет! Это Drowsiness Detection Bot.
Отправьте картинку c изображением людей.''')


@dp.message(F.photo)
async def echo_photo(message: types.Message):
    dest = f"C:/Users/amine/PycharmProjects/KABYTE_BOT/static/photo.jpg"
    await bot.download(
        message.photo[-1],
        destination=dest
    )

    image = Image.open(dest)
    if await classify_person(image, message.chat.id):
        await message.answer_photo(FSInputFile("output.png"))


async def main():
    await dp.start_polling(bot)


asyncio.run(main())
