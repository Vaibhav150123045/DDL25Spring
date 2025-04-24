Set COUNTER=0
:x

echo %Counter%
if "%Counter%"=="6" (
    echo "END!"
) else (
    timeout /t 1
    start cmd.exe /c "python B2_exercise.py %Counter%"
    set /A COUNTER+=1
    goto x
)
