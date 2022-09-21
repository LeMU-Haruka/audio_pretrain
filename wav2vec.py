import os

import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio


































model_type = {
    'base': torchaudio.pipelines.WAV2VEC2_BASE,
    'ft_10M': torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_base = torchaudio.pipelines.WAV2VEC2_BASE
model = wav2vec_ft.get_model().to(device)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


torch.random.manual_seed(0)


print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SPEECH_FILE = "./_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)

wav2vec_ft = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

labels = wav2vec_ft.get_labels()

wav2vec_base = torchaudio.pipelines.WAV2VEC2_BASE

print("Sample Rate:", wav2vec_ft.sample_rate)

print("Labels:", labels)

model = wav2vec_ft.get_model().to(device)

print(model.__class__)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

# if sample_rate != bundle.sample_rate:
#     waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
#
# with torch.inference_mode():
#     features, _ = model.extract_features(waveform)
#
# fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
# for i, feats in enumerate(features):
#     ax[i].imshow(feats[0].cpu())
#     ax[i].set_title(f"Feature from transformer layer {i+1}")
#     ax[i].set_xlabel("Feature dimension")
#     ax[i].set_ylabel("Frame (time-axis)")
# plt.tight_layout()
# plt.show()

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

with torch.inference_mode():
    emission, _ = model(waveform)

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.show()
print("Class labels:", labels)

decoder = GreedyCTCDecoder(labels=labels)
transcript = decoder(emission[0])

print(transcript)


def load_wav2vec():
    torchaudio.models.wav2vec2_base()
