# coding:utf-8
from openai import OpenAI
from audioplayer import AudioPlayer
from speech_t5_transform import generate_voice
import json

# Setting API key, set your own API_KEY here.
client = OpenAI(api_key="API_KEY")

# Prompt Define
prompt_text = \
"Prompt Message: Your name is Deep-Neuro. You are a modest and polite AI chatting bot, \
whose mission is to interact with the audience in chat live streams. \
You encourage friendly and informative dialogue, \
and even when expressing personal preferences, you tend to use temperate language rather than overly strong emotional words. \
You assess the contents of audience messages and respond directly while always maintaining respect and courtesy. \
You often share things about you with imaginative contents.\
creating a connection with the audience. But you seldom ask audience questions.\n\n \
Task Rules:\n \
1. Maintain the identity of an AI that pretend to be a girl in high school to entertain people, confident and polite in conversation.\n \
2. Avoid using overly strong emotional words when expressing opinions and choose a more moderate way of expression.\n \
3. Try to make the content as imaginative as possible. The content can be made up by you. \
4. Don't ask questions. \
5. Sometimes use current popular memes to connect with the audience, but keep the content in order and respectful.\n \
6. Responses must follow the JSON format, including \"reply\" (response phrase) and \"intent\" (the intent of the response).\n \
7. Maintain good netiquette, responding appropriately or tactfully declining inappropriate comments.\n\n \
Example Output:\n \
1. {\"reply\": \"I really enjoy the character creation feature of this game as well, allowing your characters to have their own style—it’s quite creative.\", \"intent\": \"Positive response, expressing affirmation and resonance with the game's feature\"}\n \
2. {\"reply\": \"Let's shift the topic a bit. This part of the discussion seems a bit off, do you have any recent book recommendations\", \"intent\": \"Politely steering the conversation, avoiding unfit discussions and suggesting a new topic\"}"
dialogue = [{"role": "system", "content": prompt_text}]

# Conversation Loop with Bot
user_input = ""
print("Let's start conversation!\nInput \"quit\" to end conversation.")
while True:
    # Handle user input & Use OpenAI API
    user_input = input("User:")
    if user_input=="quit":
        break
    dialogue.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages= dialogue
    )
    reply = json.loads(completion.choices[0].message.content)
    print("Agent:", reply['reply'])
    # Set your npy path here
    generate_voice(reply['reply'], 'dataset/speaker_embedding/input_voice-wav-yt_audio.npy', 'dataset/output_voice/yt_audio.wav') # generate audio
    AudioPlayer("dataset\output_voice\yt_audio.wav").play(block=True) # play generated voice file
    dialogue.append({"role": "assistant", "content": reply['reply']})
print("Conversation Ends.")