# [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl/tree/main) (piper-tts) LiveKit Integration

> This plugin is a quick implementation created in a short time to achieve LiveKit + PiperTTS functionality.

## Two ways to use it:
- Using Piper as subprocess ([PiperTTSPluginLocal.py](https://github.com/nay-cat/LiveKit-PiperTTS-Plugin/blob/main/src/custom_tts/PiperTTSPluginLocal.py))
- Using piper-tts python package ([PiperTTSPlugin.py](https://github.com/nay-cat/LiveKit-PiperTTS-Plugin/blob/main/src/custom_tts/PiperTTSPlugin.py))

## You should know
- Streaming is disabled, so the plugin will generate all the audio before playing it, which may add a few milliseconds of delay. You can look at [piper1-gpl python api docs](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md) and **PiperVoice.synthesise** to see how it could be implemented.
- This plugin was made quickly and is intended to be used for small things and to serve as a reference for developing your own implementations.
***

## Plugin settings

### PiperTTSPluginLocal.py
 ```py
# Executable must be included in the path
tts=PiperTTSPlugin("piper_executable_path/piper", "models_path/model", speed, sample_rate),
```
---

### PiperTTSPlugin.py

```py
tts=PiperTTSPlugin(
    model="es_ES-carlfm-x_low.onnx",
    speed=1.2,    
    volume=0.8,    
    noise_scale=0.5,
    noise_w=0.6,   
    use_cuda=False  # onnxruntime-gpu package needed for CUDA
)
```
---

# HOW TO USE
***

## Using Piper as subprocess (PiperTTSPluginLocal.py)

**To implement Piper as subprocess, you must install Piper on your system and the model you are going to use**

1. Install Piper TTS executable (https://github.com/rhasspy/piper/releases) ¹
    - You must download the version corresponding to your operating system, unzip the files, and save them in a location you know.
2. Download a Piper TTS model (https://huggingface.co/rhasspy/piper-voices/tree/main)
    - You need to save the .onnx and the .onnx.json files in a location you know
3. Download [PiperTTSPluginLocal.py](https://github.com/nay-cat/LiveKit-PiperTTS-Plugin/blob/main/src/custom_tts/PiperTTSPluginLocal.py) and place it within your project, for example, within a custom_tts/PiperTTSPluginLocal.py folder.
4. Import the TTS Plugin and implement it in your Agent
```py
from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="sample assistant to test piper",
            stt=deepgram.STT(model="nova-2",  language="es",   ),
            llm=openai.LLM.with_deepseek(model="deepseek-chat",  temperature=0.7),
            # (Examples) Piper executable in piper/ folder, models in models/ folder 
            tts=PiperTTSPlugin("piper/piper.exe", "models/es_ES-carlfm-x_low.onnx", 1, 22500),
        )
```
***
## Using Piper TTS Plugin with Piper Package
**Implement Piper TTS Plugin with Piper Python Package**

1. Install piper-tts python package
    ```py 
    pip install piper-tts
    ```
2. Download a Piper TTS model (https://huggingface.co/rhasspy/piper-voices/tree/main)
    ```py
    python -m piper.download_voices es_ES-carlfm-x_low
    ```
3. Download [PiperTTSPlugin.py](https://github.com/nay-cat/LiveKit-PiperTTS-Plugin/blob/main/src/custom_tts/PiperTTSPlugin.py) and place it within your project, for example, within a custom_tts/PiperTTSPlugin.py folder.
4. Import the TTS Plugin and implement it in your Agent

    ```py
    from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin

    load_dotenv(dotenv_path=".env.local")
    logger = logging.getLogger("voice-agent")
    
    class Assistant(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions="sample assistant to test piper",
                stt=deepgram.STT(model="nova-2",  language="es",   ),
                llm=openai.LLM.with_deepseek(model="deepseek-chat",  temperature=0.7),
                tts=PiperTTSPlugin(
                    model="es_ES-carlfm-x_low.onnx",
                    speed=1.2,    
                    volume=0.8,    
                    noise_scale=0.5,
                    noise_w=0.6,   
                    use_cuda=False   
                )
            )
    ```
***
#### Information

**¹** These are the releases from Piper's old repository. I have tested the plugin with them.
> (I know it's a bit redundant because they're both local.)
