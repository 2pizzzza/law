import json
import easyocr
from fuzzywuzzy import fuzz


def main():
    path_img = f'Autofill_fin/for_detecting/3.jpg'
    text = easyocr.Reader(["ru", "en"]).readtext(path_img, detail=0, paragraph=False, text_threshold=0.5)

    data = [easyocr.Reader(["ru", "en"]).readtext(f'Autofill_fin/for_detecting/{i}.jpg', detail=0, paragraph=False,
                                                  text_threshold=0.8) for i in range(9) if i != 3]

    data_fields = {
        0: ['Birthdate'],
        1: ['Expiration'],
        2: ['Document'],
        3: ['Rus Name'],
        4: ['Eng Name'],
        5: ['Nationality'],
        6: ['Patronymic'],
        7: ['Sex'],
        8: ['Rus Surname'],
        9: ['Eng Surname'],
    }

    counter = 0

    for field in data:
        for part in field:
            for s in text:
                if fuzz.ratio(part, s) > 60:
                    if counter >= 10:
                        break
                    data_fields[counter].append(text[text.index(s)])
                    counter += 1

    ready_data = {}
    for i in range(len(data_fields)):
        ready_data[data_fields[i][0]] = data_fields[i][1]

    json_data = json.dumps(ready_data, ensure_ascii=False)
    return json_data


print(main())
