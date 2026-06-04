param(
    [string]$Python = "C:\Users\ADMIN\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

& $Python -m pip install -r requirements.txt
& $Python -m PyInstaller --noconfirm --clean --onefile CameraAsciiQuad.spec