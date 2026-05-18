# Student 03 – C2 Model Testing

## Model principal
Gemini 2.5 Flash Lite

## Model de rezerva
OpenRouter Free

## Temperatura recomandata
0.1

## Motivare
Modelul Gemini 2.5 Flash Lite ofera raspunsuri consistente, corecte si stabile.
Respecta instructiunile si genereaza JSON valid, fiind potrivit pentru adnotare.

OpenRouter este utilizat ca fallback deoarece ofera o alternativa disponibila in cazul limitelor de API, desi prezinta erori JSON si instabilitate.

## Ethics and Safety Checklist

Aplicația include un strat simplu de siguranță și etică pentru simulările politice generate de AI.

Măsuri implementate:
- disclaimer pentru conținut politic generat de AI;
- scor de risc pentru limbaj manipulator;
- detectarea expresiilor speculative sau conspiraționiste;
- transparență privind contextul recuperat prin RAG;
- suport pentru agenți cu perspective ideologice diferite.

Scopul aplicației este simularea educațională și analiza comportamentului agenților AI, nu oferirea de recomandări politice sau informații factuale garantate.