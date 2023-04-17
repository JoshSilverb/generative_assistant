# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
# import spacy
# import torch
import requests
import json


weather_API_key = "f2130b95abcdbcaf650457d30eb0c2ce"

label_names = {0: "restaurant_reviews", 1: "nutrition_info", 2: "account_blocked", 3: "oil_change_how", 4: "time", 5: "weather", 6: "redeem_rewards", 7: "interest_rate", 8: "gas_type", 9: "accept_reservations", 10: "smart_home", 11: "user_name", 12: "report_lost_card", 13: "repeat", 14: "whisper_mode", 15: "what_are_your_hobbies", 16: "order", 17: "jump_start", 18: "schedule_meeting", 19: "meeting_schedule", 20: "freeze_account", 21: "what_song", 22: "meaning_of_life", 23: "restaurant_reservation", 24: "traffic", 25: "make_call", 26: "text", 27: "bill_balance", 28: "improve_credit_score", 29: "change_language", 30: "no", 31: "measurement_conversion", 32: "timer", 33: "flip_coin", 34: "do_you_have_pets", 35: "balance", 36: "tell_joke", 37: "last_maintenance", 38: "exchange_rate", 39: "uber", 40: "car_rental", 41: "credit_limit", 42: "oos", 43: "shopping_list", 44: "expiration_date", 45: "routing", 46: "meal_suggestion", 47: "tire_change", 48: "todo_list", 49: "card_declined", 50: "rewards_balance", 51: "change_accent", 52: "vaccines", 53: "reminder_update", 54: "food_last", 55: "change_ai_name", 56: "bill_due", 57: "who_do_you_work_for", 58: "share_location", 59: "international_visa", 60: "calendar", 61: "translate", 62: "carry_on", 63: "book_flight", 64: "insurance_change", 65: "todo_list_update", 66: "timezone", 67: "cancel_reservation", 68: "transactions", 69: "credit_score", 70: "report_fraud", 71: "spending_history", 72: "directions", 73: "spelling", 74: "insurance", 75: "what_is_your_name", 76: "reminder", 77: "where_are_you_from", 78: "distance", 79: "payday", 80: "flight_status", 81: "find_phone", 82: "greeting", 83: "alarm", 84: "order_status", 85: "confirm_reservation", 86: "cook_time", 87: "damaged_card", 88: "reset_settings", 89: "pin_change", 90: "replacement_card_duration", 91: "new_card", 92: "roll_dice", 93: "income", 94: "taxes", 95: "date", 96: "who_made_you", 97: "pto_request", 98: "tire_pressure", 99: "how_old_are_you", 100: "rollover_401k", 101: "pto_request_status", 102: "how_busy", 103: "application_status", 104: "recipe", 105: "calendar_update", 106: "play_music", 107: "yes", 108: "direct_deposit", 109: "credit_limit_change", 110: "gas", 111: "pay_bill", 112: "ingredients_list", 113: "lost_luggage", 114: "goodbye", 115: "what_can_i_ask_you", 116: "book_hotel", 117: "are_you_a_bot", 118: "next_song", 119: "change_speed", 120: "plug_type", 121: "maybe", 122: "w2", 123: "oil_change_when", 124: "thank_you", 125: "shopping_list_update", 126: "pto_balance", 127: "order_checks", 128: "travel_alert", 129: "fun_fact", 130: "sync_device", 131: "schedule_maintenance", 132: "apr", 133: "transfer", 134: "ingredient_substitution", 135: "calories", 136: "current_location", 137: "international_fees", 138: "calculator", 139: "definition", 140: "next_holiday", 141: "update_playlist", 142: "mpg", 143: "min_payment", 144: "change_user_name", 145: "restaurant_suggestion", 146: "travel_notification", 147: "cancel", 148: "pto_used", 149: "travel_suggestion", 150: "change_volume"}
weekday_to_num = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday':6}
num_to_weekday = {0: 'monday', 1: 'tuesday', 2: 'wednesday', 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}


print("Loading intent tokenizer...")
# intent_tokenizer = AutoTokenizer.from_pretrained("transformersbook/distilbert-base-uncased-distilled-clinc")
print("Loading intent classifier model...")
# intent_model = AutoModelForSequenceClassification.from_pretrained("transformersbook/distilbert-base-uncased-distilled-clinc")
print("Loading text generation model...")
# text_generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
print("Loading spacy model...")
# spacy_ner = spacy.load("en_core_web_sm")
print("Done")





def get_intent(text):
    tokenized_text = intent_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = intent_model(**tokenized_text).logits

    predicted_class_id = logits.argmax().item()

    return predicted_class_id, label_names[predicted_class_id]


def get_keywords(text):
    doc = spacy_ner(text)
    kwds = {}
    for ent in doc.ents:
        if ent.label_ in kwds:
            kwds[ent.label_].append(ent.text)
        else:
            kwds[ent.label_] = [ent.text]
    
    return kwds



def handle_weather(keywords):
    if 'GPE' not in keywords or 'DATE' not in keywords:
        return {'status': 'failed', 'error': 'Location or date not in keywords'}

    # assume only one location and date
    location = keywords['GPE'][0].lower()
    date = keywords['DATE'][0].lower()

    today = datetime.now()
    weekday = today.weekday() # gives num value of day of week
    
    
    if date == 'today':
        days_from_now = 0
    elif date == 'tomorrow':
        days_from_now = 1
    elif date in weekday_to_num:
        if weekday_to_num[date] < weekday: # eg currently Thurs, user asks about Tues
            days_from_now = (7 - weekday) + weekday_to_num[date]
        else: # eg currently Tues, user asks about Thurs
            days_from_now = weekday_to_num[date] - weekday
    else: # if something unexpected just run for today
        days_from_now = 0

    # get weather data for next 5 days in 3hr intervals
    r = requests.get("http://api.openweathermap.org/data/2.5/forecast", params={'q': location, 'appid': weather_API_key})
    response = r.json()
    cnt = response['cnt'] # number of data points (should be 40 = 50days / 3hr increments)
    print(f'cnt={cnt}')
    weather_data = {'max_temp': [0,0,0,0,0], 'min_temp': [0,0,0,0,0], 'feels_like': [0,0,0,0,0], 'pop': [0,0,0,0,0], 'description': ['','','','','']}

    # average out data day by day
    for day in range(int(cnt/8)): 
        descs = []
        for it in range(8):
            block = response['list'][5*day+it]

            weather_data["max_temp"][day] += block['main']['temp_max']
            weather_data["min_temp"][day] += block['main']['temp_min']
            weather_data["feels_like"][day] += block['main']['feels_like']
            weather_data["pop"][day] += block['pop']
            descs.append(block['weather'][0]['description'])
        
        # average out weather data over the course of the day
        weather_data["max_temp"][day] /= 8
        weather_data["min_temp"][day] /= 8
        weather_data["feels_like"][day] /= 8
        weather_data["pop"][day] /= 8

        # loop thru descriptions to find mode
        counts = {}
        for d in descs:
            if d in counts:
                counts[d] += 1
            else:
                counts[d] = 1
        print(counts)
        weather_data["description"][day] = max(counts, key=counts.get)
    
    print(weather_data)

    on_the_day = {'max_temp': k_to_f(weather_data['max_temp'][days_from_now]), 'min_temp': k_to_f(weather_data['min_temp'][days_from_now]), 'pop': int(100 * weather_data['pop'][days_from_now]), 'description': weather_data['description'][days_from_now]}

    weather_out = {'status': 'success', 'text': f"The weather in {location} on {num_to_weekday[days_from_now]} is {on_the_day['description']}, with a high of {on_the_day['max_temp']}°F, low of {on_the_day['min_temp']}°F, a {on_the_day['pop']}% chance of precipitation"}
    
    return weather_out


def k_to_f(k):
    return int((k - 273.15) * (9/5) + 32)



def data_to_text(query, data):
    # prompt = query + " given the following data: " + json.dumps(data)
    # output = text_generator(prompt, do_sample=True, min_length=50)
    # return output[0]['generated_text']

    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(API_URL, headers=headers, json={"inputs": query + " given the following data: " + json.dumps(data)})
    print(response)
    return response.json()



if __name__ == "__main__":
    
    input_text = "What's the weather like in Ann Arbor tomorrow?"

    # print("Running intent model...")
    # intent_id, intent = get_intent(input_text)
    # print(f"Got intent={intent} with id={intent_id}")

    # print("Running keywords NER model...")
    # keywords = get_keywords(input_text)
    # print(f"Got keywords: {keywords}")
    
    intent = "weather"
    keywords = {'GPE': ['Ann Arbor'], 'DATE': ['tomorrow']}

    result = handle_weather(keywords)
    print(result)
    exit(0)

    result = {}
    if intent == 'weather':
        print("Getting weather info")
        result = handle_weather(keywords)
        print(f"Got weather info: {result}")
    else:
        result = {'status': 'failed', 'error': f'Unsupported intent detected: {intent}'}



    if result['status'] != 'failed':
        print("Running text gen model...")
        output_text = data_to_text(input_text, result)
        print(f"text gen model output={output_text}")
    
    else:
        print(result)
        exit(1)