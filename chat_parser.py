import re
import pandas as pd
from urlextract import URLExtract
import emoji

def parse_chat(file):
    lines = file.read().decode('utf-8').split('\n')
    data = []

    pattern1 = r'^\[(\d{1,2})/(\d{1,2})/(\d{2}),\s(\d{1,2}):(\d{2}):(\d{2})[\s\u202f]([AP]M)\]\s([~]?[^:]+?):\s*(.*)$'
    pattern2 = r'^(\d{1,2})/(\d{1,2})/(\d{2,4}),\s(\d{1,2}):(\d{2})\s([AP]M)\s-\s([^:]+?):\s*(.*)$'

    for line in lines:
        match1 = re.match(pattern1, line)
        if match1:
            day, month, year_short, hour, minute, second, ampm, sender, message = match1.groups()
            year_long = f'20{year_short}'
            date = f'{day}/{month}/{year_long}'
            time = f'{hour}:{minute}:{second} {ampm}'
            data.append([date, time, sender.strip(), message.strip()])
        else:
            match2 = re.match(pattern2, line)
            if match2:
                day, month, year, hour, minute, ampm, sender, message = match2.groups()
                date = f'{day}/{month}/{year}'
                time = f'{hour}:{minute} {ampm}'
                data.append([date, time, sender.strip(), message.strip()])
            elif 'end-to-end encrypted' not in line and 'image omitted' not in line and line.strip():
                if data:
                    data[-1][3] += '\n' + line.strip()

    df = pd.DataFrame(data, columns=['date', 'time', 'sender', 'message'])
    return df

def extract_features(df, user):
    from datetime import datetime

    user_df = df[df['sender'] == user].copy()
    extractor = URLExtract()

    urls = user_df['message'].apply(extractor.find_urls).sum()
    emojis = user_df['message'].apply(lambda msg: [c for c in msg if c in emoji.EMOJI_DATA]).sum()
    word_counts = user_df['message'].apply(lambda x: len(x.split()))
    avg_word_count = word_counts.mean()
    msg_count = user_df.shape[0]
    emoji_count = len(emojis)
    url_count = len(urls)

    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed', dayfirst=True, errors='coerce')
    df = df.sort_values('timestamp').copy()
    df['prev_sender'] = df['sender'].shift(1)
    df['prev_time'] = df['timestamp'].shift(1)

    user_responses = df[(df['sender'] == user) & (df['prev_sender'] != user)].copy()
    user_responses.loc[:, 'response_time'] = (user_responses['timestamp'] - user_responses['prev_time']).dt.total_seconds()
    avg_response_time_seconds = user_responses['response_time'].mean()
    avg_response_time_minutes = avg_response_time_seconds / 60 if not pd.isna(avg_response_time_seconds) else 9999

    return {
        'msg_count': msg_count,
        'avg_word_count': avg_word_count,
        'emoji_count': emoji_count,
        'url_count': url_count,
        'avg_response_time': avg_response_time_minutes
    }