# Syncron

![BPM Analyzer](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*(Русская версия ниже)*

A highly precise CLI tool for detecting the exact BPM (Beats Per Minute) of an audio file. It leverages multiple independent algorithms and combines their results to achieve high accuracy.

**Note:** This program works best for static BPM. If you have a track with a variable BPM, well, you'll have to find it yourself manually 😅.

## Features
- **Multi-Algorithm Fusion:** Uses `librosa`'s dynamic programming beat tracker, autocorrelation on the onset envelope, and a Fourier tempogram.
- **Sub-Pixel Accuracy:** Employs parabolic interpolation to find the exact peak frequencies, granting accuracy up to two decimal places.
- **Smart Consensus:** Automatically normalizes octave ambiguities (e.g. 70 vs 140 BPM) and scores candidates to find the most probable tempo.
- **Visual Plots:** Generates a beautiful summary plot (Waveform, Onset Strength, Fourier Tempogram, Mel Spectrogram) and saves it to the `plots/` directory with the name of the original song.
- **Stability Check:** Analyzes segments of the song to see if the tempo drifts or remains stable.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MianderOfficial/Syncron
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

Basic analysis (uses all algorithms):
```bash
python bpm_analyzer.py song.mp3
```

Only analyze the first 60 seconds (much faster):
```bash
python bpm_analyzer.py song.mp3 --duration 60
```

Generate visualizations (saved in `plots/` folder):
```bash
python bpm_analyzer.py song.mp3 --plot
```

See the output of all internal algorithms and candidates:
```bash
python bpm_analyzer.py song.mp3 --all
```

Start from a specific offset:
```bash
python bpm_analyzer.py song.mp3 --start 30 --duration 60
```

### Important Note on Files
Place your music files (e.g., `.mp3`, `.wav`) in the **same directory** as the `bpm_analyzer.py` script for easiest access.

### Why "Syncron"?
The name **Syncron** is a play on *Synchronization*. It represents the tool's ability to "sync" perfectly with the underlying rhythm of any audio. It's about finding that hidden BPM and bringing it to light with technical precision.

---

#Syncron

*(English version above)*

Высокоточная консольная утилита для определения точного BPM (ударов в минуту) аудиофайла. Она использует несколько независимых алгоритмов и объединяет их результаты для достижения максимальной точности.

**Примечание:** Эта программа отлично работает в основном для статичного BPM. Переменный BPM вам придётся находить самому ручками :) 

## Функционал
- **Комбинирование алгоритмов:** Использует поиск долей через динамическое программирование (`librosa`), автокорреляцию onset-огибающей и Фурье-темпограмму.
- **Субпиксельная точность:** Применяет параболическую интерполяцию для поиска точных пиковых частот (до сотых долей).
- **Умный консенсус:** Автоматически решает проблему октав (например, 70 против 140 BPM) и выбирает наиболее вероятный темп.
- **Графики:** Создает красивый график анализа (Waveform, Onset Strength, Fourier Tempogram, Mel Spectrogram) и сохраняет его в папку `plots/` под названием оригинального трека.
- **Проверка стабильности:** Анализирует фрагменты песни, чтобы определить, плавает ли темп или остается стабильным.

## Установка

1. Склонируйте репозиторий:
```bash
git clone https://github.com/MianderOfficial/Syncron
```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Как использовать

Базовый анализ:
```bash
python bpm_analyzer.py song.mp3
```

Анализ только первых 60 секунд (работает намного быстрее):
```bash
python bpm_analyzer.py song.mp3 --duration 60
```

Сгенерировать графики (сохраняются в папку `plots/`):
```bash
python bpm_analyzer.py song.mp3 --plot
```

Посмотреть результаты всех алгоритмов детально:
```bash
python bpm_analyzer.py song.mp3 --all
```

Начать анализ с определенной секунды:
```bash
python bpm_analyzer.py song.mp3 --start 30 --duration 60
```

### Важное примечание по файлам
Сохраняйте музыкальные файлы (например, `.mp3`, `.wav`) в **папке с проектом**, там же, где находится файл программы `bpm_analyzer.py`, для максимально удобного использования.

### Почему "Syncron"?
Название **Syncron** происходит от слова *Синхронизация*. Оно отражает способность инструмента идеально «синхрониться» с внутренним ритмом любого аудио. Это про поиск того самого скрытого БПМа и выведение его на свет с технической точностью.
