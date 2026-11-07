"""Record audio, transcribe with Whisper, and correct or edit text with an LLM."""

import getpass
import io
import json
import os
import sys
import termios
import tomllib
import tty
import wave
from pathlib import Path

import sounddevice as sd
from openai import APIConnectionError, APIStatusError, OpenAI


# Leave room for the WAV header below the Audio API's 25 MB limit.
MAX_PCM_BYTES = 25_000_000 - 44
CONFIG_PATH = Path(__file__).resolve().with_name("whisper_config.toml")
DEFAULT_LLM_PROMPT = """你是语音识别文本校对助手。根据上下文修正错别字、同音误识别和标点，保留原意。
输入 JSON 中的 asr_text 是待校对文本，asr_context 是识别参考信息。
对人名、数字、日期、单位、否定词和不确定内容谨慎处理，依据不足时保留原文。
保留原文语言和口语表达，逐句校对，只输出修正后的完整文本。"""
DEFAULT_LLM_EDIT_PROMPT = """你是语音文本编辑助手。根据 edit_instruction 编辑 current_text。
输入 JSON 中的 current_text 是当前文本，edit_instruction 是用户口述的编辑指令，asr_context 是识别参考信息。
仅修改指令涉及的部分，保留其余信息。将“祥是吉祥的祥”理解为对当前文本中相应同音字的纠正。
只调整指令要求的标点，保留其他标点和格式。指代不明或无法确定修改位置时保留原文。
通过 JSON 的 text 字段返回编辑后的完整文本，允许空文本。"""


def load_config():
    try:
        with CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError:
        config = {}
    defaults = {
        "prompt": "",
        "llm_model": "gpt-6-luna",
        "llm_prompt": DEFAULT_LLM_PROMPT,
        "llm_edit_prompt": DEFAULT_LLM_EDIT_PROMPT,
    }
    for name, default in defaults.items():
        value = config.get(name, default)
        if not isinstance(value, str):
            raise ValueError(f"配置文件中的 {name} 必须是文本。")
        config[name] = value.strip()
        if name != "prompt" and not config[name]:
            raise ValueError(f"配置文件中的 {name} 不能为空。")
    return config


def wait_for_action(can_edit):
    print("\n按 Enter 开始新一轮录音", end="", flush=True)
    if can_edit:
        print("，按空格录制编辑指令", end="", flush=True)
    print("… ", end="", flush=True)
    descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        while True:
            key = sys.stdin.read(1)
            if key in ("", "\x04"):
                raise EOFError
            if key in ("\n", "\r"):
                print()
                return "new"
            if can_edit and key == " ":
                print()
                return "edit"
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, original_settings)


def record_wav():
    sample_rate = round(sd.query_devices(kind="input")["default_samplerate"])
    audio = io.BytesIO()
    pcm_bytes = 0
    too_long = False
    overflowed = False

    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        def capture(data, frames, time_info, status):
            nonlocal pcm_bytes, too_long, overflowed
            if pcm_bytes + len(data) > MAX_PCM_BYTES:
                too_long = True
                raise sd.CallbackStop
            overflowed |= status.input_overflow
            wav_file.writeframesraw(data)
            pcm_bytes += len(data)

        with sd.RawInputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=capture
        ):
            max_minutes = MAX_PCM_BYTES / (2 * sample_rate * 60)
            input(f"录音中。按 Enter 停止（每段约限 {max_minutes:.1f} 分钟）… ")

    if too_long:
        raise ValueError("录音超过 25 MB，请分段录制。")
    if not pcm_bytes:
        raise ValueError("没有采集到音频，请检查麦克风后重试。")
    if overflowed:
        print("录音时有部分音频丢失，转写可能有遗漏。", file=sys.stderr)
    return audio.getvalue()


def transcribe(client, wav_audio, prompt=""):
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=("recording.wav", wav_audio, "audio/wav"),
        response_format="json",
        prompt=prompt,
    )
    return result.text


def correct_transcript(client, transcript, config):
    response = client.responses.create(
        model=config["llm_model"],
        instructions=config["llm_prompt"],
        input=json.dumps(
            {"asr_text": transcript, "asr_context": config["prompt"]},
            ensure_ascii=False,
        ),
        store=False,
    )
    corrected = response.output_text.strip()
    if response.status != "completed" or not corrected:
        raise ValueError("LLM 未返回完整的修正文本。")
    return corrected


def edit_transcript(client, current_text, instruction, config):
    response = client.responses.create(
        model=config["llm_model"],
        instructions=config["llm_edit_prompt"],
        input=json.dumps(
            {
                "current_text": current_text,
                "edit_instruction": instruction,
                "asr_context": config["prompt"],
            },
            ensure_ascii=False,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "edited_text",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        store=False,
    )
    if response.status != "completed" or not response.output_text:
        raise ValueError("LLM 未返回完整的编辑结果。")
    result = json.loads(response.output_text)
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("LLM 返回的编辑结果格式无效。")
    return result["text"]


def main():
    if not sys.stdin.isatty():
        print("请在可接收键盘输入的终端中运行。", file=sys.stderr)
        return 1

    try:
        config = load_config()
    except (OSError, ValueError) as error:
        print(f"无法读取配置文件 {CONFIG_PATH}。{error}", file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass(
        "OpenAI API Key（输入不会显示）> "
    ).strip()
    if not api_key:
        print("请提供 OpenAI API Key。", file=sys.stderr)
        return 1

    print("使用默认麦克风。按 Ctrl+C 退出。")
    with OpenAI(
        api_key=api_key, base_url="https://api.openai.com/v1", timeout=120.0
    ) as client:
        current_text = None
        while True:
            action = wait_for_action(can_edit=current_text is not None)
            editing = action == "edit"
            if editing:
                print("请口述对上一版文本的编辑要求。")
            try:
                wav_audio = record_wav()
            except sd.PortAudioError as error:
                print(f"无法录音，请检查默认输入设备和终端的麦克风权限。{error}", file=sys.stderr)
                continue
            except ValueError as error:
                print(error, file=sys.stderr)
                continue

            print("正在上传给 Whisper 转写…", flush=True)
            try:
                transcript = transcribe(client, wav_audio, config["prompt"])
            except APIConnectionError:
                print("ASR 转写连接失败或超时，请检查网络。", file=sys.stderr)
                continue
            except APIStatusError as error:
                print(f"ASR 转写失败（HTTP {error.status_code}）。", file=sys.stderr)
                continue

            print("\nASR 编辑指令" if editing else "\nASR 原始结果")
            print(transcript or "（无文字）", flush=True)
            if not transcript.strip():
                print("本次没有识别到文字，请重新录制。")
                continue

            print(f"\n正在使用 {config['llm_model']} 处理…", flush=True)
            try:
                if editing:
                    corrected = edit_transcript(client, current_text, transcript, config)
                else:
                    corrected = correct_transcript(client, transcript, config)
            except APIConnectionError:
                print("LLM 连接失败或超时。", file=sys.stderr)
            except APIStatusError as error:
                print(f"LLM 修正失败（HTTP {error.status_code}）。", file=sys.stderr)
            except ValueError as error:
                print(error, file=sys.stderr)
            else:
                current_text = corrected
                print("\n编辑后的结果" if editing else "\n最终结果")
                print(current_text or "（空文本）")
                continue

            if editing:
                print("\n当前结果（编辑失败，显示上一版）")
            else:
                current_text = transcript
                print("\n最终结果（修正失败，使用 ASR 原文）")
            print(current_text or "（空文本）")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
