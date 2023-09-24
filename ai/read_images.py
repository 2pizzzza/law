import os.path
import cv2
import easyocr
from fuzzywuzzy import fuzz
from PIL import Image


def main():
    Eto_ID = False
    path_img = f'../../../PycharmProjects/pythonProject2/Autofill_fin/for_detecting/3.jpg'
    text = easyocr.Reader(["ru", "en"]).readtext(path_img, detail=0, paragraph=False, text_threshold=0.5)

    ID_CARD = ['ИДЕНТИФИКАЦИЯЛЫК КАРТА', 'ИДЕНТИФИКАЦИОННАЯ КАРТА', 'IDENTITY CARD']

    # Check if it is ID Card
    for i in ID_CARD:
        for s in text:
            if fuzz.ratio(i, s) > 50 or i == s:
                Eto_ID = True

    print(text)

    # Surname = ['Фамилиясы /', '/ Фамилия /', '/ Surname']
    Surname = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/8.jpg', detail=0, paragraph=False, text_threshold=0.8)
    print(Surname)
    # Name = ['Аты /', '/ Имя /', '/ Name']
    Name = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/4.jpg', detail=0, paragraph=False, text_threshold=0.8)
    print(Name)
    # Patronymic = ['Атасынын аты /', '/ Отчество /', '/ Patronymic']
    Patronymic = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/6.jpg', detail=0, paragraph=False, text_threshold=0.8)
    print(Patronymic)
    # Sex = ['Жынысы /', '/ Пол /', '/ Sex']
    Sex = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/7.jpg', detail=0, paragraph=False, text_threshold=0.8)
    # Nationality = ['Жарандыгы', '/ Гражданство /', '/ Nationality']
    Nationality = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/5.jpg', detail=0, paragraph=False, text_threshold=0.8)
    # Birthdate = ['Туулган куну /', '/ Дата рождения /', '/ Date of birth']
    Birthdate = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/0.jpg', detail=0, paragraph=False, text_threshold=0.8)
    # Document = ['Документтин /', '/ документа /', '/ Document']
    Document = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/2.jpg', detail=0, paragraph=False, text_threshold=0.8)
    # Expirition = ['Колдонуу мооноту', 'Срок действия', '/ Date of expiry']
    Expirition = easyocr.Reader(["ru", "en"]).readtext('Autofill_fin/for_detecting/1.jpg', detail=0, paragraph=False, text_threshold=0.8)
    
  
    for i in Surname:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Eng_Surname_id = text.index(s) + 1
                Rus_Surname_id = text.index(s)
                Surname = [text[Eng_Surname_id], text[Rus_Surname_id]]
                break
            
    for i in Name:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Eng_Name_id = text.index(s) + 1
                Rus_Name_id = text.index(s)
                Name = [text[Eng_Name_id], text[Rus_Name_id]]
            
    for i in Patronymic:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Rus_Patronymic_id = text.index(s) + 1
                Patronymic = text[Rus_Patronymic_id]
            
    for i in Sex:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Sex_id = text.index(s)
                Sex = text[Sex_id]
                # if preSex == 'M':
                #     Sex = 'Э/М'
                # else:
                #     Sex = 'К/Ж'
            
    for i in Nationality:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Nationality_id = text.index(s)
                Nationality = text[Nationality_id]
            
    for i in Birthdate:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Birthdate_id = text.index(s)
                Birthdate = text[Birthdate_id]
            
    for i in Document:
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Document_id = text.index(s)
                Document = text[Document_id]
    
    for i in Expirition:    
        for s in text:
            if fuzz.ratio(i, s) > 60:
                Expirition_id = text.index(s)
                Expirition = text[Expirition_id]


    print('Surname: ', Surname) 
    print('Name:', Name)   
    print('Patronymic: ',Patronymic)
    print('Sex: ', Sex)
    print('Nationality: ', Nationality)
    print('Birthdate: ', Birthdate)
    print('Document: ', Document)
    print('Expirition: ', Expirition)
    
main()