@echo off
chcp 65001 >nul
title Демонстрация лабораторных работ по компьютерной графике

:menu
cls
echo ========================================================
echo   Компьютерная графика реального времени (МАИ)
echo   Меню демонстрации лабораторных работ
echo ========================================================
echo.
echo   [1] Лабораторная 1: Каркас Vulkan и треугольник (ветка master)
echo   [2] Лабораторная 2: Камера, преобразования, Blinn-Phong (ветка lab2)
echo   [3] Лабораторная 3: Текстурирование и сэмплирование (ветка lab3)
echo   [4] Лабораторная 4: Shadow Mapping и Dynamic Rendering (ветка lab4)
echo.
echo   [0] Выход
echo.
echo ========================================================
set /p choice="Выберите номер лабораторной [0-4]: "

if "%choice%"=="1" (
    call "%~dp0run_lab1.bat"
    goto menu
)
if "%choice%"=="2" (
    call "%~dp0run_lab2.bat"
    goto menu
)
if "%choice%"=="3" (
    call "%~dp0run_lab3.bat"
    goto menu
)
if "%choice%"=="4" (
    call "%~dp0run_lab4.bat"
    goto menu
)
if "%choice%"=="0" (
    exit /b 0
)
echo Неверный выбор! Попробуйте снова.
timeout /t 2 >nul
goto menu
