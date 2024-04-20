from fastapi import FastAPI, HTTPException,UploadFile,File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import imutils
import easyocr
import requests

app = FastAPI()

class ImagePath(BaseModel):
    image_path: str


@app.post("/api/performocr")
async def perform_ocr(image: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary location
        with open("temp_image.jpg", "wb") as temp_image:
            temp_image.write(await image.read())

        # Perform OCR on the saved image
        result = ocr("temp_image.jpg")
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
def ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) # Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    number = result[0][-2]

    url = "https://rto-vehicle-information-india.p.rapidapi.com/getVehicleInfo"

    payload = {
        "vehicle_no": number,
        "consent": "Y",
        "consent_text": "I hereby give my consent for Eccentric Labs API to fetch my information"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "6030fee68amshdb66fd4d11353e0p11ebc1jsnaf8da4a93548",
        "X-RapidAPI-Host": "rto-vehicle-information-india.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)


    json_data =response.json()
    registration_authority = json_data['data']['registration_authority']
    registration_no = json_data['data']['registration_no']
    registration_date = json_data['data']['registration_date']
    owner_name = json_data['data']['owner_name']
    fuel_type = json_data['data']['fuel_type']
    vehicle_class = json_data['data']['vehicle_class']
    vehicle_info = json_data['data']['vehicle_info']
    
    
    return{
        "Registration Authority": registration_authority,
        "Registration Number": registration_no,
        "Registration Date":registration_date,
        "Owner Name": owner_name,
        "Fuel Type":fuel_type,
        "Vehicle Class": vehicle_class,
        "Vehicle Info ": vehicle_info
    }
