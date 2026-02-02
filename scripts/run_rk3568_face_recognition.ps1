$Python = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe"

& $Python -m pip install -r requirements.txt

# Examples:
# Collect (requires explicit YES confirmation in the app)
# & $Python rk3568_face_recognition\app.py collect --name "user1" --camera-index 0
#
# Train
# & $Python rk3568_face_recognition\app.py train
#
# Recognize
& $Python rk3568_face_recognition\app.py recognize --camera-index 0


