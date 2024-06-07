from uuid import uuid4

from flask import Flask, render_template, request, jsonify
import requests
import json
import re


BOT_TOKEN = "NOT SHOWN"
BOT_ID = 7185070968
app = Flask(__name__, template_folder='.')
global last_update_id

@app.route('/', methods=['POST'])
def handle_webhook():
    try:
        process(json.loads(request.get_data()))
        return 'Success!'
    except Exception as e:
        print(e)
        return 'Error'

def random():
    global last_update_id
    last_update_id = -1
    while True:
        updates = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_update_id}&allowed_updates=["message","inline_query","callback_query"]').json().get('result', [])
        for update in updates:
            process(update)
            last_update_id = update['update_id'] + 1

def process(update):
    if 'message' in update:
        user_id = str(update['message']['from']['id'])
        with open('users.txt', 'r') as file:
            users = file.readlines()
            found = False
            name = 'Guest'
            for user in users:
                if user.split()[0] == user_id.strip():
                    name = user.split()[2:]
                    found = True
                    break
            if not found:
                print(update)
                if 'text' in update['message'] and update['message']['text'] == 'Log in':
                    inline_keyboard = {"inline_keyboard": [[{"text": "Select your account", "switch_inline_query_current_chat": ""}]]}
                    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',json={'chat_id': user_id, 'text': f'_Find your name and select!_','reply_markup': inline_keyboard, 'parse_mode': 'Markdown'})
                elif 'via_bot' in update['message'] and update['message']['via_bot']['id'] == BOT_ID:
                    text = update['message']['text'][5:]
                    reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"}, {'text': "3", 'callback_data': "3"}], [{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"}, {'text': "6", 'callback_data': "6"}], [{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"}, {'text': "9", 'callback_data': "9"}], [{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"}, {'text': "Show 🐵", 'callback_data': "show"}], [{'text': "Submit ✅", 'callback_data': "submit"}]]}
                    requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',json={'chat_id': user_id, 'text': f'Logging in as {text}\nPassword: ','reply_markup': reply_markup, 'parse_mode': 'Markdown'})
                else:
                    keyboard = {'keyboard': [['Log in']], 'one_time_keyboard': False, 'resize_keyboard': True}
                    data = {'chat_id': user_id, 'text': f'Please log in first!', 'reply_markup': json.dumps(keyboard),'parse_mode': 'Markdown'}
                    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data)
                return
        if 'via_bot' in update['message'] and update['message']['via_bot']['id'] == BOT_ID:
            requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage', json={'chat_id': user_id, 'text': "*Action can not be fulfilled!*\n_You need to log out first!_", 'parse_mode': 'Markdown'})
        if 'text' in update['message']:
            message = update['message']['text']
            if message == '/start':
                keyboard = {
                    'keyboard': [['Profile', 'Stats'], ['Log out', 'Notification']],
                    'one_time_keyboard': False,
                    'resize_keyboard': True
                }
                data = {'chat_id': user_id, 'text': f'Successfully logged in as *{" ".join(name)}*', 'reply_markup': json.dumps(keyboard),'parse_mode': 'Markdown'}
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data)
            elif message == 'Stats':
                text = '<strong>Your Statistics:\n\n</strong>'
                with open(f"{'_'.join(name)}.txt", 'r') as file:
                    random = file.readline()
                    lines = file.readlines()
                    for line in lines:
                        text += line
                requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',json={'chat_id': user_id, 'text': text,'parse_mode': 'HTML'})
            elif message == 'Profile':
                with open(f"{'_'.join(name)}.jpg", 'rb') as file:
                    reply_markup = {'inline_keyboard': [[{'text': "Update Your Photo", 'callback_data': f"update"}, {'text': "Update Your Password", 'callback_data': f"change"}], [{'text': "Delete Your Account", 'callback_data': f"delete"}]]}
                    print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto',data={'chat_id': 5934725286, 'caption': '_' + ' '.join(name) + '_ - ' + ' Student at *New Uzbekistan University*', 'protect_content': True, 'reply_markup': json.dumps(reply_markup), 'parse_mode': 'Markdown'}, files={'photo': file}).json())
            elif message == 'Notification':
                with open(f"{'_'.join(name)}.txt", 'r') as file:
                    choice = file.readline()
                    if choice.strip() == 'Y':
                        reply_markup = {'inline_keyboard': [[{'text': "Turn off notification", 'callback_data': f"off"}]]}
                        data = {'chat_id': user_id, 'text': f'*You have enabled attendance notification! 🔊*','reply_markup': json.dumps(reply_markup), 'parse_mode': 'Markdown'}
                    else:
                        reply_markup = {'inline_keyboard': [[{'text': "Turn on notification", 'callback_data': f"on"}]]}
                        data = {'chat_id': user_id, 'text': f'*You have disabled attendance notification! 🔇*','reply_markup': json.dumps(reply_markup), 'parse_mode': 'Markdown'}
                    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data)
            elif message == 'Log out':
                new_lines = ''
                with open('users.txt', 'r') as file:
                    users = file.readlines()
                    for user in users:
                        print(user.split()[0])
                        if user_id == user.split()[0]:
                            new_lines += '0' + " " + ' '.join(user.split()[1:]) + '\n'
                            print('0' + " " + ' '.join(user.split()[1:]))
                        else:
                            new_lines += user
                    with open('users.txt', 'w') as file:
                        file.writelines(new_lines)
                keyboard = {'keyboard': [['Log in']], 'one_time_keyboard': False, 'resize_keyboard': True}
                requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',json={'chat_id': user_id, 'text': "*Logged out successfully!*\n_You can always log in at any time!_", 'reply_markup': json.dumps(keyboard), 'parse_mode': 'Markdown'})
    elif 'photo' in update:
        pass
    elif 'inline_query' in update:
        with open('users.txt', 'r') as file:
            lines = file.readlines()

        # Filter articles based on the query
        articles = [line.split()[2:] for line in lines if update['inline_query']['query'].lower() in ' '.join(line.split()[2:]).lower()]
        print("length:", articles)
        # Sort filtered articles by popularity (assuming the second column represents popularity)
        # filtered_articles = sorted(articles, key=lambda article: int(article[1]), reverse=True)

        # Parse the offset from the update, default to 0 if not provided
        offset = int(update['inline_query']['offset']) if update['inline_query']['offset'] and update['inline_query']['offset'] != 'null' else 0

        # Calculate next offset
        next_offset = str(offset + 20) if offset + 20 < len(lines) else ''

        # Construct results for inline query response
        results = []
        for article in articles[offset:offset + 20]:
            # Construct result item for each article
            result_item = {'id': str(uuid4()),  # Generate a unique ID for the result item
                'title': ' '.join(article),  # Title of the article
                'type': 'article',  # Type of the result item
                'input_message_content': {  # Content of the message when the result is selected
                    'message_text': "I am " + ' '.join(article),  # Text of the message
                    'parse_mode': 'Markdown'  # Parse mode (optional)
                }}
            # Append result item to results list
            results.append(result_item)

        # Send the response to the inline query
        response = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/answerInlineQuery",
            json={'inline_query_id': update['inline_query']['id'], 'results': json.dumps(results), 'next_offset': next_offset,'cache_time': 20}, headers={'Content-Type': 'application/json'})
        print(response.json())
    elif 'callback_query' in update:
        print(update)
        data = update['callback_query']['data']
        user_id = update['callback_query']['from']['id']
        try:
            text = update['callback_query']['message']['text']
        except:
            print('no text')
        try:
            password = update['callback_query']['message']['reply_markup']["inline_keyboard"][3][2]['callback_data'][4:]
        except:
            print('no password')
        message_id = update['callback_query']['message']['message_id']
        if data.isdigit():
            if password == '':
                password = int(data)
            else:
                password = 10 * int(password) + int(data)
            print(password)
            text = re.sub(r'\d', '*', text)
            text += "*"
            reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"},{'text': "3", 'callback_data': "3"}],[{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"},{'text': "6", 'callback_data': "6"}],[{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"},{'text': "9", 'callback_data': "9"}],[{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"},{'text': "Show 🐵", 'callback_data': f"show{password}"}], [{'text': "Submit ✅", 'callback_data': "submit"}]]}
            print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id,'text': text, 'reply_markup': json.dumps(reply_markup),'parse_mode': 'HTML'}).json())
        elif data[:4] == 'show':
            text = text.rstrip('*')
            text += password
            reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"},{'text': "3", 'callback_data': "3"}],[{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"},{'text': "6", 'callback_data': "6"}],[{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"},{'text': "9", 'callback_data': "9"}],[{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"},{'text': "Hide 🙈", 'callback_data': f"hide{password}"}], [{'text': "Submit ✅", 'callback_data': "submit"}]]}
            print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id,'text': text, 'reply_markup': json.dumps(reply_markup),'parse_mode': 'HTML'}).json())
        elif data[:4] == 'hide':
            text = re.sub(r'\d', '*', text)
            print(text)
            reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"},{'text': "3", 'callback_data': "3"}],[{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"},{'text': "6", 'callback_data': "6"}],[{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"},{'text': "9", 'callback_data': "9"}],[{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"},{'text': "Show 🐵", 'callback_data': f"show{password}"}], [{'text': "Submit ✅", 'callback_data': "submit"}]]}
            print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id,'text': text, 'reply_markup': json.dumps(reply_markup),'parse_mode': 'HTML'}).json())
        elif data == 'clear':
            if text[-1] != ':':
                text = text[:-1]
                password = int(int(password) / 10)
                reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"},{'text': "3", 'callback_data': "3"}],[{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"},{'text': "6", 'callback_data': "6"}],[{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"},{'text': "9", 'callback_data': "9"}],[{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"},{'text': "Show 🐵", 'callback_data': f"show{password}"}], [{'text': "Submit ✅", 'callback_data': "submit"}]]}
                print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id,'text': text, 'reply_markup': json.dumps(reply_markup),'parse_mode': 'HTML'}).json())
            else:
                pass
                ##################### asnwer querry saying are you dumb?
        elif data == 'submit':
            print(password)
            newline_index = text.find('\n')
            name = text[14:newline_index]
            new_lines = ''
            with open('users.txt', 'r') as file:
                users = file.readlines()
                cont = True
                for user in users:
                    print(' '.join(user.split()[2:]))
                    if name == ' '.join(user.split()[2:]):
                        if password == user.split()[1]:
                            print(user.split()[1])
                            keyboard = {'keyboard': [['Profile', 'Stats'], ['Log out', 'Notification']],'one_time_keyboard': False, 'resize_keyboard': True}
                            data = {'chat_id': user_id, 'text': f'Successfully logged in as *{name}*','reply_markup': json.dumps(keyboard), 'parse_mode': 'Markdown'}
                            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data)
                            new_lines += str(user_id) + " " + ' '.join(user.split()[1:]) + '\n'
                            cont = False
                            print(str(user_id) + " " + ' '.join(user.split()[1:]) + '\n')
                        else:
                            print('yemadi')
                            ##################### asnwer querry saying are you dumb?
                            return
                    if cont:
                        new_lines += user
                    else:
                        cont = True
                with open('users.txt', 'w') as file:
                    file.writelines(new_lines)
            reply_markup = {'inline_keyboard': [[{'text': "1", 'callback_data': "1"}, {'text': "2", 'callback_data': "2"},{'text': "3", 'callback_data': "3"}],[{'text': "4", 'callback_data': "4"}, {'text': "5", 'callback_data': "5"},{'text': "6", 'callback_data': "6"}],[{'text': "7", 'callback_data': "7"}, {'text': "8", 'callback_data': "8"},{'text': "9", 'callback_data': "9"}],[{'text': "0", 'callback_data': "0"}, {'text': "Clear ◀️", 'callback_data': "clear"},{'text': "Show 🐵", 'callback_data': f"show{password}"}],[{'text': "Submit ✅", 'callback_data': "submit"}]]}
            print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id, 'text': text,'reply_markup': json.dumps(reply_markup), 'parse_mode': 'HTML'}).json())
        elif data == 'update':
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",json={'callback_query_id': update['callback_query']['id'], 'text': "We are working on this...",'show_alert': False})
        elif data == 'change':
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",json={'callback_query_id': update['callback_query']['id'], 'text': "We are working on this...",'show_alert': False})
        elif data == 'delete':
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",json={'callback_query_id': update['callback_query']['id'], 'text': "We are working on this...",'show_alert': False})
        elif data == 'on' or data == 'off':
            user_id = str(update['callback_query']['from']['id'])
            with open('users.txt', 'r') as file:
                users = file.readlines()
                name = 'Guest'
                for user in users:
                    if user.split()[0] == user_id.strip():
                        name = user.split()[2:]
                        with open(f"{'_'.join(name)}.txt", 'r') as file:
                            random = file.readline()
                            lines = file.readlines()
                        if random.strip() == 'Y':
                            random = 'N\n'
                            text = f'*You have disabled attendance notification! 🔇*'
                            reply_markup = {'inline_keyboard': [[{'text': "Turn on notification", 'callback_data': f"on"}]]}
                        else:
                            random = 'Y\n'
                            text = f'*You have enabled attendance notification! 🔊*'
                            reply_markup = {'inline_keyboard': [[{'text': "Turn off notification", 'callback_data': f"off"}]]}
                        break
            with open(f"{'_'.join(name)}.txt", 'w') as file:
                print(random)
                file.write(random)
                file.writelines(lines)
            print(requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/editMessageText',json={'chat_id': user_id, 'message_id': message_id,'reply_markup': reply_markup, 'text': text, 'parse_mode': 'Markdown'}).json())
    else:
        print(update)


if __name__ == '__main__':
    #random()
    app.run(host='0.0.0.0', port=5000)
