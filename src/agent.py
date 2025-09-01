import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
)
from livekit.plugins import (
    openai,
    deepgram,
    silero,
)

from livekit.plugins import noise_cancellation
from custom_tts.PiperTTSPluginLocal import PiperTTSPlugin



load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="sample assistant to test piper",
            stt=deepgram.STT(
                model="nova-2",  
                language="es",   
            ),
            llm=openai.LLM.with_deepseek(
                model="deepseek-chat",  
                temperature=0.7
            ),
            tts=PiperTTSPlugin("piper/piper.exe", "models/es_ES-carlfm-x_low.onnx", 1, 22500),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )

    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )