# Transform a YouTube video to text using either direct transcript API or audio-to-text fallback
# Modified to include youtube-transcript-api as primary method

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None
import whisper
from langdetect import detect
from pytubefix import YouTube


# Function to create and open a txt file
def create_and_open_txt(text, filename):
    # Create and write the text to a txt file
    with open(filename, "w") as file:
        file.write(text)


def transcribe_youtube_video(url):
    """
    Transcribe a YouTube video either using direct transcript API or audio-to-text fallback.

    Args:
        url (str): YouTube video URL

    Returns:
        str: Transcribed text
    """
    # Fallback to audio-to-text method
    yt = YouTube(url)

    # Get the audio stream
    audio_stream = yt.streams.filter(only_audio=True).first()

    # Download the audio stream
    output_path = "YoutubeAudios"
    filename = "audio.mp3"
    audio_stream.download(output_path=output_path, filename=filename)

    print(f"Audio downloaded to {output_path}/{filename}")

    # Load the base model and transcribe the audio
    model = whisper.load_model("large")
    result = model.transcribe("YoutubeAudios/audio.mp3")
    transcribed_text = result["text"]
    print(transcribed_text)

    # Detect the language
    language = detect(transcribed_text)
    print(f"Detected language: {language}")

    # Create and open a txt file with the text
    # create_and_open_txt(transcribed_text, f"output_{language}.txt")
    return transcribed_text


if __name__ == "__main__":
    # Ask user for the YouTube video URL
    url = input("Enter the YouTube video URL: ")
    transcribe_youtube_video(url)
