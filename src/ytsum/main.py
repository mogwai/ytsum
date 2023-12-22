import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from io import BytesIO
from transformers.utils import is_flash_attn_2_available
from pytube import YouTube


def transcribe(audio):

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # audio = audio.to(torch_dtype)
    model_id = "openai/whisper-large-v2"
    device = "cuda:0"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        use_flash_attention_2=is_flash_attn_2_available(),
    )

    model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        model_kwargs={"use_flash_attention_2": is_flash_attn_2_available()},
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        batch_size=1,
        device=device,
    )

    return pipe(audio, generate_kwargs={"language": "en"})

def download_youtube_audio_bytes(url):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    buffer = BytesIO()
    stream.stream_to_buffer(buffer)
    buffer.seek(0)
    return buffer

def main():
    print("Using Flash Attn", is_flash_attn_2_available())
    url = "https://www.youtube.com/watch?v=Htg3HCgrJK4"
    audio = download_youtube_audio_bytes(url)
    audio, sr = torchaudio.load(audio)
    torchaudio.save("out.wav", audio, sr)
    audio = audio.sum(0)[:sr*5*60]
    print(audio.shape, sr)
    text = transcribe(audio.numpy())
    print(text)
    # parse with whisper

    # Summarise with LLM

if __name__ == "__main__":
    main()
