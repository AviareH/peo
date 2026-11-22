# Robot vision

Laptop-side camera processing and path estimation for PEO CV. 99% vibecoded

## Connect to the ESP32

1. Power the ESP32.
2. Connect the laptop to **PEO-Camera**, password **peocamera123**.
3. Close any browser tab viewing the camera stream.
4. Run from the repository root:

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream --preview
```

For semantic detection and depth with cached models:

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream --method semantic --model b0 --depth --offline --preview
```

Press **Q** to exit. Results save under `vision/outputs`.
