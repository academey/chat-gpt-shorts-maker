import openai
import requests
from PIL import Image
from io import BytesIO
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips, ColorClip, vfx
import json
import re
from pydub import AudioSegment

import os
from gtts import gTTS


# OpenAI API 키 설정
openai.api_key = 'api-key'

def generate_script(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a storytelling assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

prompt = """
재밌는 사실 8가지를 알려줘.

동영상의 본문은 논의 중인 주제에 대한 추가 설명이 들어갑니다. 추가 설명에는 최초 진술에 덧붙일 과학적 사실과 역사적 증거에 대한 것이 포함됩니다 대본에는 이야기 내용만 포함되어야 하며 카메라 촬영 및 장면 전환과 같은 추가적인 스크립트 기능은 포함되지 않습니다. 각 문장은 구체적이어야 하고, 간결해야 하며 전문적인 언어를 사용해야 합니다
"""

script = generate_script(prompt)
# script="test"
# print(script)

# 이미지 프롬프트 생성 프롬프트
image_prompt_template = """
응답필드는 항상 영어로 대답해라
대본을 6개의 장면 단위로 나누고, 각 장면마다 사용할 사진 두 개를 정하고, 사진에 대한 설명과 나레이터가 사용할 보이스오버를 JSON 형식으로 작성해주세요. JSON 형식은 다음과 같아야 합니다:

[
  {
    "scene_number": 1,
    "images": [
      {"description": "사진 1 설명", "text": "사진 1 텍스트"},
      {"description": "사진 2 설명", "text": "사진 2 텍스트"}
    ],
    "voiceover": "보이스오버 텍스트"
  },
  ...
  {
    "scene_number": 6,
    "images": [
      {"description": "사진 1 설명", "text": "사진 1 텍스트"},
      {"description": "사진 2 설명", "text": "사진 2 텍스트"}
    ],
    "voiceover": "보이스오버 텍스트"
  }
]

대본:
""" + script
def generate_image_prompts():
    prompt = image_prompt_template
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "한글로 대답해라"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

image_prompts_text = generate_image_prompts()
print("Generated Image Prompts Text:", image_prompts_text)

# JSON 형식 부분만 추출하는 함수
def extract_json(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# JSON 형식 추출 및 파싱
json_text = extract_json(image_prompts_text)
if json_text:
    image_prompts = json.loads(json_text)
    print("Parsed Image Prompts:", image_prompts)
else:
    print("No valid JSON found in the response.")
    exit(1)
# image_prompts = [
#   {
#     "scene_number": 1,
#     "images": [
#       {"description": "식물이 환경 변화에 반응하는 장면", "text": "식물이 화학물질을 방출하며 소통하는 모습"},
#       {"description": "숲 속 식물들이 상호작용하는 장면", "text": "다양한 종의 식물이 서로 상호작용하며 소통하는 모습"}
#     ],
#     "voiceover": "과학적 연구에 따르면, 식물도 인간과 마찬가지로 환경 변화에 반응하여 소통합니다. 이러한 식물 간 소통은 우리에게는 믿기 어려울 수 있지만, 사실입니다."
#   },
#   {
#     "scene_number": 2,
#     "images": [
#       {"description": "식물들이 환경 정보를 공유하는 장면", "text": "식물들이 환경 정보를 교환하며 상호작용하는 모습"},
#       {"description": "숲 속 식물들이 협력하는 장면", "text": "식물들이 서로를 도와가며 더 나은 환경을 조성하는 모습"}
#     ],
#     "voiceover": "식물들은 서로를 위해 환경 정보를 공유하고 함께 협력하여 더 나은 환경을 만들어냅니다. 이는 식물들의 놀라운 소통 능력을 보여줍니다."
#   },
#   {
#     "scene_number": 3,
#     "images": [
#       {"description": "식물 감각을 발휘하는 장면", "text": "식물이 주변 환경을 감지하고 반응하는 모습"},
#       {"description": "식물 세계의 복잡성을 담은 장면", "text": "다양한 종류의 식물이 복잡하게 상호작용하는 모습"}
#     ],
#     "voiceover": "식물들이 환경 감지 능력을 통해 서로 상호작용하며 복잡한 관계를 형성하는 것을 볼 때, 우리는 식물 세계의 풍부함을 느낄 수 있습니다."
#   },
#   {
#     "scene_number": 4,
#     "images": [
#       {"description": "식물 통역사의 중요성을 강조하는 장면", "text": "식물 통역사가 식물의 신호를 해석하는 모습"},
#       {"description": "식물 커뮤니케이션의 신비한 세계를 탐색하는 장면", "text": "식물들이 다양한 방식으로 소통하고 교류하는 모습"}
#     ],
#     "voiceover": "식물 통역사의 역할은 환경 보호와 지속 가능한 미래를 위해 중요합니다. 식물의 커뮤니케이션은 우리에게 신비한 세계를 엿보게 합니다."
#   },
#   {
#     "scene_number": 5,
#     "images": [
#       {"description": "식물의 소통 능력을 강조하는 장면", "text": "식물들이 서로의 의사를 소통하는 모습"},
#       {"description": "식물 세계의 다양성을 보여주는 장면", "text": "다양한 형태와 크기의 식물들이 함께 존재하는 모습"}
#     ],
#     "voiceover": "식물들은 서로의 의사를 소통하며 다양성을 표현합니다. 식물 세계는 사람들이 상상하는 것 이상으로 다채롭고 신비한 곳입니다."
#   },
#   {
#     "scene_number": 6,
#     "images": [
#       {"description": "식물 세계의 교훈과 발견을 상징하는 장면", "text": "식물들이 함께 존재하며 우리에게 교훈을 전하는 모습"},
#       {"description": "식물과 사람의 상호작용을 담은 장면", "text": "사람이 식물과 소통하며 상호작용하는 모습"}
#     ],
#     "voiceover": "식물 세계는 우리에게 무한한 교훈과 발견을 선사합니다. 우리는 식물과 함께 살아가며 존중과 이해를 배울 수 있습니다."
#   }
# ]

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='ko')
    tts.save(output_file)
    print(f"Audio saved to {output_file}")
    # pydub을 사용하여 오디오 속도 변경
    sound = AudioSegment.from_file(output_file)
    sound = sound.speedup(playback_speed=1.5)  # 1.5배 속도
    sound.export(output_file, format="mp3")
    print(f"Audio speed adjusted and saved to {output_file}")

audio_file_korean = "output_audio.mp3"
text_to_speech(script, audio_file_korean)

def generate_image(prompt, img_number):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    image_path = f"image_{img_number}.png"

    # 이미지 다운로드 및 저장
    response_image = requests.get(image_url)
    with open(image_path, 'wb') as f:
        f.write(response_image.content)

    return image_path
def split_text_in_half(text):
    words = text.split()
    mid_point = len(words) // 2
    return ' '.join(words[:mid_point]), ' '.join(words[mid_point:])
def split_text(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(0, len(words), n)]

# 이미지 생성 및 비디오 클립 생성
image_paths = []
audio_paths = []

clips = []
for i, scene in enumerate(image_prompts):
    part1, part2 = split_text_in_half(scene['voiceover'])

    # 각각의 텍스트로 TTS 생성
    audio_file_path_part1 = f"voiceover_{i}_part1.mp3"
    audio_file_path_part2 = f"voiceover_{i}_part2.mp3"
    text_to_speech(part1, audio_file_path_part1)
    text_to_speech(part2, audio_file_path_part2)
    audio_paths.extend([audio_file_path_part1, audio_file_path_part2])

    # 음성 파일 클립 생성
    audio_clip_part1 = AudioFileClip(audio_file_path_part1)
    audio_clip_part2 = AudioFileClip(audio_file_path_part2)

    for j, img_desc in enumerate(scene['images']):
        img_path = generate_image(img_desc['text'], f"{i}_{j}")
        # img_path = f"image_{i}_{j}.png"
        image_paths.append(img_path)

        # 이미지 클립 생성
        if j == 0:
            image_clip = ImageClip(img_path).set_duration(audio_clip_part1.duration)
            audio_clip = audio_clip_part1
            voiceover_text = part1
        else:
            image_clip = ImageClip(img_path).set_duration(audio_clip_part2.duration)
            audio_clip = audio_clip_part2
            voiceover_text = part2

        # 이미지 확대 효과 추가
        # image_clip = image_clip.fx(vfx.resize, lambda t: 1 + 0.01 * t)

        # 텍스트 클립 생성
        split_texts = split_text(voiceover_text, 2)
        text_clips = []
        for idx, txt in enumerate(split_texts):
            # # 텍스트 배경 클립 생성
            txt_clip = TextClip(txt, fontsize=70, font='KoreanGD19R', color='white')
            bg_clip = ColorClip(size=(txt_clip.w, 120), color='black')
            bg_clip = bg_clip.set_position((1, 10), relative=True)

            txt_clip = CompositeVideoClip([bg_clip, txt_clip.set_position(("center", 0), relative=True)])
            txt_clip = txt_clip.set_start(idx * (audio_clip.duration / len(split_texts))).set_duration(
                audio_clip.duration / len(split_texts)).set_position(("center", 500))
            text_clips.append(txt_clip)
            # 텍스트 배경과 텍스트 결합

            # txt_clip = TextClip(txt, fontsize=70, font='KoreanGD19R', color='white')
            # txt_shadow = TextClip(txt, fontsize=70, font='KoreanGD19R', color='black')
            # txt_shadow = txt_shadow.set_position((5, 5), relative=True)
            # txt_clip = CompositeVideoClip([txt_shadow, txt_clip.set_position((0, 0), relative=True)])
            # txt_clip = txt_clip.set_start(idx * (audio_clip.duration / len(split_texts))).set_duration(
            #     audio_clip.duration / len(split_texts)).set_position(("center", 500))
            # text_clips.append(txt_clip)

        # 이미지와 텍스트를 결합한 클립 생성
        video_clip = CompositeVideoClip([image_clip] + text_clips).set_duration(audio_clip.duration).crossfadein(0.03)
        video_clip = video_clip.set_audio(audio_clip)
        clips.append(video_clip)

# 모든 클립을 하나의 비디오로 결합
final_clip = concatenate_videoclips(clips, method="compose")

# 음성 파일 추가
# audio_clip = AudioFileClip(audio_file_korean)
# if audio_clip.duration > final_clip.duration:
#     audio_clip.set_duration(final_clip.duration)
# else:
#     final_clip.set_duration(audio_clip.duration)
# final_clip = final_clip.set_audio(audio_clip)

# 비디오 파일로 저장
output_path = "final_video.mp4"
final_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

# # 생성된 이미지 및 음성 파일 삭제 (정리)
# for image_path in image_paths:
#     os.remove(image_path)
#
# for audio_path in audio_paths:
#     os.remove(audio_path)

print(f"Video saved to {output_path}")