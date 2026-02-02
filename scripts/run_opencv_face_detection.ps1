$Python = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python38\python.exe"

& $Python -m pip install -r requirements.txt

# Webcam (ESC to quit)
& $Python opencv_face_detection\detect_faces.py --camera 0


