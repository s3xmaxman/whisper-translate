import argparse
from faster_whisper import WhisperModel
from datetime import timedelta
import os
import time

# config
task = "transcribe"


def format_time(seconds):
    """
    秒数をHH:MM:SS,mmm形式の文字列に変換する。
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def transcribe_video(input_file):
    """
    ビデオファイルをトランスクリブし、SRTファイルを生成する。

    引数:
    input_file (str): トランスクリビングするビデオファイルのパス。
    """
    model_size = "distil-large-v3"  # English Only model
    print(f"Loading model '{model_size}'...")
    start_time = time.time()
    model = WhisperModel(model_size, device="cpu", cpu_threads=12, compute_type="int8")
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")

    print("Starting transcription...")
    segments, info = model.transcribe(
        input_file, beam_size=5, task=task, vad_filter=True
    )

    print(
        "Detected language '{}' with probability {:.2f}".format(
            info.language, info.language_probability
        )
    )

    srt_filename = os.path.splitext(input_file)[0] + ".srt"

    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for segment in segments:
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            text = segment.text
            segment_id = segment.id + 1
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            print(line_out)
            srt_file.write(
                f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            )
            srt_file.flush()


def get_srt_files():
    """現在のディレクトリ内のsrtファイルのリストを取得する。

    Returns:
        list: srtファイルのリスト
    """
    return [f for f in os.listdir(".") if f.endswith(".srt")]


def select_srt_file():
    """ユーザーにsrtファイルを選択させる。

    Returns:
        str: 選択されたsrtファイル名
    """
    srt_files = get_srt_files()
    if not srt_files:
        print("No .srt files found in the current directory.")
        return None

    print("Available .srt files:")
    for i, file in enumerate(srt_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("Enter the number of the file you want to use: "))
            if 1 <= choice <= len(srt_files):
                return srt_files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """
    メイン関数。コマンドライン引数を解析し、トランスクリビングを開始する。
    引数がない場合は、ユーザーにファイルを選択させる。
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio from a video file and generate an SRT file."
    )
    parser.add_argument(
        "input_file", nargs="?", help="Path to the video file for transcription"
    )

    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    else:
        input_file = select_srt_file()

    if input_file:
        transcribe_video(input_file)
    else:
        print("No file selected. Exiting.")


if __name__ == "__main__":
    main()
