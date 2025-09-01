import numpy as np
import asyncio
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit import rtc
from piper import PiperVoice, SynthesisConfig

# https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
class PiperTTSPlugin(tts.TTS):
    def __init__(self, model, speed=1.0, volume=1.0, noise_scale=0.667, noise_w=0.8, use_cuda=False):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=22050, num_channels=1)
        self.model = model
        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.use_cuda = use_cuda
        self._voice = None
        self._load_voice()

    def _load_voice(self):
        # according to the docs if you enable cuda you need onnxruntime-gpu package, read the docs
        self._voice = PiperVoice.load(self.model, use_cuda=self.use_cuda)
        
    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return PiperApiStream(self, text, conn_options)

class PiperApiStream(tts.ChunkedStream):
    def __init__(self, plugin, text, conn_options):
        super().__init__(tts=plugin, input_text=text, conn_options=conn_options)
        self.plugin = plugin

    async def _run(self):
        try:
            config = SynthesisConfig(
                volume=self.plugin.volume,
                length_scale=self.plugin.speed,
                noise_scale=self.plugin.noise_scale,
                noise_w_scale=self.plugin.noise_w,
                normalize_audio=True
            )
            
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self._synthesize_chunks, config)
            
            for chunk in chunks:
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=22050,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2
                )
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id="1",
                        segment_id="1",
                        frame=frame
                    )
                )
                
        except Exception as e:
            silence = np.zeros(22050, dtype=np.int16).tobytes()
            frame = rtc.AudioFrame(
                data=silence,
                sample_rate=22050,
                num_channels=1,
                samples_per_channel=22050
            )
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id="1",
                    segment_id="1",
                    frame=frame
                )
            )
    # this is a not streaming implementation, so if you are reading this, check https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md and PiperVoice.synthesize
    # i will not add streaming support right now
    def _synthesize_chunks(self, config):
        chunks = []
        for chunk in self.plugin._voice.synthesize(self.input_text, syn_config=config):
            audio_data = chunk.audio_int16_bytes
            if chunk.sample_channels == 2:
                audio = np.frombuffer(audio_data, dtype=np.int16)
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                audio_data = audio.tobytes()
            chunks.append(audio_data)
        return chunks