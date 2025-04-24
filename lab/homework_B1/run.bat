Set COUNTER=0
:x

echo %Counter%
if "%Counter%"=="3" (
    echo "END!"
) else (
    timeout /t 1
    start cmd.exe /c "python PP_minibatch.py %Counter%"
    set /A COUNTER+=1
    goto x
)
