# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test audio_utils"""
import os
import unittest
import tempfile
import subprocess
import numpy as np
import pytest

from mindformers.dataset.transforms.audio_utils import (
    amplitude_to_db,
    amplitude_to_db_batch,
    chroma_filter_bank,
    hertz_to_mel,
    mel_filter_bank,
    mel_to_hertz,
    power_to_db,
    power_to_db_batch,
    spectrogram,
    spectrogram_batch,
    window_function,
    fram_wave,
    stft
)
from librosa.filters import chroma


tmp_path = tempfile.TemporaryDirectory().name


def require_librosa(test_case):
    """
    Decorator marking a test that requires librosa
    """
    return unittest.skipUnless(True, "test requires librosa")(test_case)


# pylint: disable=C0111
@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestAudioUtilsFunction(unittest.TestCase):
    def test_hertz_to_mel(self):
        self.assertEqual(hertz_to_mel(0.0), 0.0)
        self.assertAlmostEqual(hertz_to_mel(100), 150.48910241)

        inputs = np.array([100, 200])
        expected = np.array([150.48910241, 283.22989816])
        self.assertTrue(np.allclose(hertz_to_mel(inputs), expected))

        self.assertEqual(hertz_to_mel(0.0, "slaney"), 0.0)
        self.assertEqual(hertz_to_mel(100, "slaney"), 1.5)

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "slaney"), expected))

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "kaldi"), expected))

        with pytest.raises(ValueError):
            hertz_to_mel(100, mel_scale=None)

    def test_mel_to_hertz(self):
        self.assertEqual(mel_to_hertz(0.0), 0.0)
        self.assertAlmostEqual(mel_to_hertz(150.48910241), 100)

        inputs = np.array([150.48910241, 283.22989816])
        expected = np.array([100, 200])
        self.assertTrue(np.allclose(mel_to_hertz(inputs), expected))

        self.assertEqual(mel_to_hertz(0.0, "slaney"), 0.0)
        self.assertEqual(mel_to_hertz(1.5, "slaney"), 100)

        inputs = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "slaney"), expected))

        inputs = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "kaldi"), expected))

        with pytest.raises(ValueError):
            mel_to_hertz(100, mel_scale=None)

    def test_mel_filter_bank_shape(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm=None,
            mel_scale="htk",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="slaney",
            triangularize_in_mel_space=True,
        )
        self.assertEqual(mel_filters.shape, (513, 13))

    def test_mel_filter_bank_htk(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="htk",
        )
        # fmt: off
        expected = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.61454786, 0.0, 0.0, 0.0],
            [0.82511046, 0.17488954, 0.0, 0.0],
            [0.35597035, 0.64402965, 0.0, 0.0],
            [0.0, 0.91360726, 0.08639274, 0.0],
            [0.0, 0.55547007, 0.44452993, 0.0],
            [0.0, 0.19733289, 0.80266711, 0.0],
            [0.0, 0.0, 0.87724349, 0.12275651],
            [0.0, 0.0, 0.6038449, 0.3961551],
            [0.0, 0.0, 0.33044631, 0.66955369],
            [0.0, 0.0, 0.05704771, 0.94295229],
            [0.0, 0.0, 0.0, 0.83483975],
            [0.0, 0.0, 0.0, 0.62612982],
            [0.0, 0.0, 0.0, 0.41741988],
            [0.0, 0.0, 0.0, 0.20870994],
            [0.0, 0.0, 0.0, 0.0]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

    def test_mel_filter_bank_slaney(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="slaney",
        )
        # fmt: off
        expected = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.39869419, 0.0, 0.0, 0.0],
            [0.79738839, 0.0, 0.0, 0.0],
            [0.80391742, 0.19608258, 0.0, 0.0],
            [0.40522322, 0.59477678, 0.0, 0.0],
            [0.00652903, 0.99347097, 0.0, 0.0],
            [0.0, 0.60796161, 0.39203839, 0.0],
            [0.0, 0.20939631, 0.79060369, 0.0],
            [0.0, 0.0, 0.84685344, 0.15314656],
            [0.0, 0.0, 0.52418477, 0.47581523],
            [0.0, 0.0, 0.2015161, 0.7984839],
            [0.0, 0.0, 0.0, 0.9141874],
            [0.0, 0.0, 0.0, 0.68564055],
            [0.0, 0.0, 0.0, 0.4570937],
            [0.0, 0.0, 0.0, 0.22854685],
            [0.0, 0.0, 0.0, 0.0]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

    def test_mel_filter_bank_kaldi(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        # fmt: off
        expected = np.array(
            [[0.0000, 0.0000, 0.0000, 0.0000],
             [0.6086, 0.0000, 0.0000, 0.0000],
             [0.8689, 0.1311, 0.0000, 0.0000],
             [0.4110, 0.5890, 0.0000, 0.0000],
             [0.0036, 0.9964, 0.0000, 0.0000],
             [0.0000, 0.6366, 0.3634, 0.0000],
             [0.0000, 0.3027, 0.6973, 0.0000],
             [0.0000, 0.0000, 0.9964, 0.0036],
             [0.0000, 0.0000, 0.7135, 0.2865],
             [0.0000, 0.0000, 0.4507, 0.5493],
             [0.0000, 0.0000, 0.2053, 0.7947],
             [0.0000, 0.0000, 0.0000, 0.9752],
             [0.0000, 0.0000, 0.0000, 0.7585],
             [0.0000, 0.0000, 0.0000, 0.5539],
             [0.0000, 0.0000, 0.0000, 0.3599],
             [0.0000, 0.0000, 0.0000, 0.1756]]
        )
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected, atol=5e-5))

    def test_mel_filter_bank_slaney_norm(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm="slaney",
            mel_scale="slaney",
        )
        # fmt: off
        expected = np.array([
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [1.19217795e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.38435591e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.40387905e-03, 5.86232616e-04, 0.00000000e+00, 0.00000000e+00],
            [1.21170110e-03, 1.77821783e-03, 0.00000000e+00, 0.00000000e+00],
            [1.95231437e-05, 2.97020305e-03, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 1.81763684e-03, 1.04857612e-03, 0.00000000e+00],
            [0.00000000e+00, 6.26036972e-04, 2.11460963e-03, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 2.26505954e-03, 3.07332945e-04],
            [0.00000000e+00, 0.00000000e+00, 1.40202503e-03, 9.54861093e-04],
            [0.00000000e+00, 0.00000000e+00, 5.38990521e-04, 1.60238924e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.83458185e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.37593638e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.17290923e-04],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.58645462e-04],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

    def test_window_function(self):
        window = window_function(16, "hann")
        self.assertEqual(len(window), 16)

        # fmt: off
        expected = np.array([
            0.0, 0.03806023, 0.14644661, 0.30865828, 0.5, 0.69134172, 0.85355339, 0.96193977,
            1.0, 0.96193977, 0.85355339, 0.69134172, 0.5, 0.30865828, 0.14644661, 0.03806023,
        ])
        # fmt: on
        self.assertTrue(np.allclose(window, expected))

    # pylint: disable=W0703
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset
        retry = True
        count = 0
        success_sig = False
        while retry:
            try:
                count += 1
                if not os.path.exists(f"{tmp_path}/validation-00000-of-00001.parquet"):
                    command = f"wget -P {tmp_path} https://hf-mirror.com/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet?download=true"
                    rename_command = f"mv {tmp_path}/*validation* {tmp_path}/validation-00000-of-00001.parquet"
                    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    subprocess.run(rename_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
                success_sig = True
                retry = False
                ds = load_dataset(tmp_path)
                speech_samples = ds.sort("id")["validation"][:num_samples]["audio"]
                return [x["array"] for x in speech_samples]
            except Exception as e:
                rm_command = f"rm -rf {tmp_path}/*validation*"
                subprocess.run(rm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(f"test_audio_utils.py has error {e} when download data.")
                if count >= 3:
                    retry = False
        if not success_sig:
            raise RuntimeError(f"test_audio_utils.py has error {e} when download data.")

    def test_spectrogram_impulse(self):
        waveform = np.zeros(40)
        waveform[9] = 1.0  # impulse shifted in time

        spec = spectrogram(
            waveform,
            window_function(12, "hann", frame_length=16),
            frame_length=16,
            hop_length=4,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (9, 11))

        expected = np.array([[0.0, 0.0669873, 0.9330127, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(spec, expected))

    def test_spectrogram_batch_impulse(self):
        waveform1 = np.zeros(40)
        waveform1[9] = 1.0

        waveform2 = np.zeros(28)
        waveform2[12] = 3.0

        waveform3 = np.zeros(51)
        waveform3[26] = 4.5

        waveform_list = [waveform1, waveform2, waveform3]

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(12, "hann", frame_length=16),
            frame_length=16,
            hop_length=4,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )

        self.assertEqual(spec_list[0].shape, (9, 11))
        self.assertEqual(spec_list[1].shape, (9, 8))
        self.assertEqual(spec_list[2].shape, (9, 13))

        expected1 = np.array([[0.0, 0.0669873, 0.9330127, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        expected2 = np.array([[0.0, 0.0, 0.75, 3.0, 0.75, 0.0, 0.0, 0.0]])
        expected3 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.375, 3.375, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertTrue(np.allclose(spec_list[0], expected1))
        self.assertTrue(np.allclose(spec_list[1], expected2))
        self.assertTrue(np.allclose(spec_list[2], expected3))

    def test_spectrogram_integration_test(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.02464888, 0.04648664, 0.05872392, 0.02311783, 0.0327175,
            0.02433643, 0.01198814, 0.02055709, 0.01559287, 0.01394357,
            0.01299037, 0.01728045, 0.0254554, 0.02486533, 0.02011792,
            0.01755333, 0.02100457, 0.02337024, 0.01436963, 0.01464558,
            0.0211017, 0.0193489, 0.01272165, 0.01858462, 0.03722598,
            0.0456542, 0.03281558, 0.00620586, 0.02226466, 0.03618042,
            0.03508182, 0.02271432, 0.01051649, 0.01225771, 0.02315293,
            0.02331886, 0.01417785, 0.0106844, 0.01791214, 0.017177,
            0.02125114, 0.05028201, 0.06830665, 0.05216664, 0.01963666,
            0.06941418, 0.11513043, 0.12257859, 0.10948435, 0.08568069,
            0.05509328, 0.05047818, 0.047112, 0.05060737, 0.02982424,
            0.02803827, 0.02933729, 0.01760491, 0.00587815, 0.02117637,
            0.0293578, 0.03452379, 0.02194803, 0.01676056,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 400], expected))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertTrue(np.allclose(spec[:64, 400], expected))

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=400,
            min_frequency=20,
            max_frequency=8000,
            sampling_rate=16000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))

        spec = spectrogram(
            waveform,
            window_function(400, "povey", periodic=False),
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
            preemphasis=0.97,
            mel_filters=mel_filters,
            log_mel="log",
            mel_floor=1.1920928955078125e-07,
            remove_dc_offset=True,
        )
        self.assertEqual(spec.shape, (400, 584))

        # fmt: off
        expected = np.array([-15.94238515, -8.20712299, -8.22704352, -15.94238515,
                             -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                             -6.52463769, -7.73677889, -15.94238515, -15.94238515,
                             -15.94238515, -15.94238515, -4.18650018, -3.37195286,
                             -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                             -4.70190154, -2.4217066, -15.94238515, -15.94238515,
                             -15.94238515, -15.94238515, -5.62755239, -3.53385194,
                             -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                             -9.43303023, -8.77480925, -15.94238515, -15.94238515,
                             -15.94238515, -15.94238515, -4.2951092, -5.51585994,
                             -15.94238515, -15.94238515, -15.94238515, -4.40151721,
                             -3.95228878, -15.94238515, -15.94238515, -15.94238515,
                             -6.10365415, -4.59494697, -15.94238515, -15.94238515,
                             -15.94238515, -8.10727767, -6.2585298, -15.94238515,
                             -15.94238515, -15.94238515, -5.60161702, -4.47217004,
                             -15.94238515, -15.94238515, -15.94238515, -5.91641988])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 400], expected, atol=1e-5))

    def test_spectrogram_batch_integration_test(self):
        waveform_list = self._load_datasamples(3)

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[2].shape, (257, 1561))

        # fmt: off
        expected1 = np.array([
            0.02464888, 0.04648664, 0.05872392, 0.02311783, 0.0327175,
            0.02433643, 0.01198814, 0.02055709, 0.01559287, 0.01394357,
            0.01299037, 0.01728045, 0.0254554, 0.02486533, 0.02011792,
            0.01755333, 0.02100457, 0.02337024, 0.01436963, 0.01464558,
            0.0211017, 0.0193489, 0.01272165, 0.01858462, 0.03722598,
            0.0456542, 0.03281558, 0.00620586, 0.02226466, 0.03618042,
            0.03508182, 0.02271432, 0.01051649, 0.01225771, 0.02315293,
            0.02331886, 0.01417785, 0.0106844, 0.01791214, 0.017177,
            0.02125114, 0.05028201, 0.06830665, 0.05216664, 0.01963666,
            0.06941418, 0.11513043, 0.12257859, 0.10948435, 0.08568069,
            0.05509328, 0.05047818, 0.047112, 0.05060737, 0.02982424,
            0.02803827, 0.02933729, 0.01760491, 0.00587815, 0.02117637,
            0.0293578, 0.03452379, 0.02194803, 0.01676056,
        ])
        expected2 = np.array([
            7.61983171e-02, 1.45338190e-01, 2.63903728e+00, 7.74429535e+00,
            9.61932980e+00, 5.40767686e+00, 1.08924884e+00, 3.40908262e+00,
            3.59484250e+00, 1.68451077e+00, 5.88405873e-01, 1.17042530e+00,
            9.94803324e-01, 3.53757065e-01, 5.47699239e-01, 9.48368581e-01,
            7.17770457e-01, 2.09396633e-01, 1.77574463e-01, 2.35644731e-01,
            1.31535991e-01, 1.53539552e-02, 4.34416305e-02, 5.32897267e-02,
            4.03567305e-02, 1.41842226e-02, 2.90514538e-02, 3.36549485e-02,
            1.53516624e-02, 2.37464225e-02, 4.60092464e-02, 4.05769324e-02,
            4.82633401e-03, 4.12675364e-02, 7.13859796e-02, 6.16866566e-02,
            2.55657822e-02, 1.68923281e-02, 1.91299946e-02, 1.60033798e-02,
            1.33405095e-02, 1.52065457e-02, 1.21833352e-02, 2.25786382e-03,
            6.15358376e-03, 1.07647616e-02, 1.23051018e-02, 6.75289378e-03,
            2.71127435e-03, 1.06515263e-02, 1.18463583e-02, 7.14347935e-03,
            1.87912782e-03, 4.44236027e-03, 5.19630243e-03, 2.46666998e-03,
            1.01598645e-03, 1.21589237e-03, 1.29095500e-03, 1.07447628e-03,
            1.40218156e-03, 3.65402623e-03, 4.00592755e-03, 4.20001841e-03
        ])
        expected3 = np.array([
            0.07805249, 0.34305022, 0.55617084, 1.22475182, 1.17040678,
            0.51540532, 0.23570016, 0.06630775, 0.09017777, 0.07693192,
            0.0333643, 0.04873054, 0.04668559, 0.02384041, 0.02780435,
            0.0289717, 0.01704903, 0.0201644, 0.01700376, 0.02176975,
            0.02042491, 0.00732129, 0.00326042, 0.00245065, 0.00510645,
            0.00681892, 0.00739329, 0.00551437, 0.0070674, 0.00630015,
            0.00379566, 0.0060098, 0.00311543, 0.00902284, 0.01171038,
            0.01202166, 0.01759194, 0.01652899, 0.01201872, 0.01295351,
            0.00756432, 0.01415318, 0.02349972, 0.02296833, 0.02429341,
            0.02447459, 0.01835044, 0.01437871, 0.02262246, 0.02972324,
            0.03392252, 0.03037546, 0.01116927, 0.01555062, 0.02833379,
            0.02294212, 0.02069847, 0.02496927, 0.02273526, 0.01341643,
            0.00805407, 0.00624943, 0.01076262, 0.01876003
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:64, 400], expected1))
        self.assertTrue(np.allclose(spec_list[1][:64, 400], expected2))
        self.assertTrue(np.allclose(spec_list[2][:64, 400], expected3))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[2].shape, (257, 1561))
        self.assertTrue(np.allclose(spec_list[0][:64, 400], expected1))
        self.assertTrue(np.allclose(spec_list[1][:64, 400], expected2))
        self.assertTrue(np.allclose(spec_list[2][:64, 400], expected3))

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=400,
            min_frequency=20,
            max_frequency=8000,
            sampling_rate=16000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "povey", periodic=False),
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
            preemphasis=0.97,
            mel_filters=mel_filters,
            log_mel="log",
            mel_floor=1.1920928955078125e-07,
            remove_dc_offset=True,
        )
        self.assertEqual(spec_list[0].shape, (400, 584))
        self.assertEqual(spec_list[1].shape, (400, 480))
        self.assertEqual(spec_list[2].shape, (400, 1247))

        # fmt: off
        expected1 = np.array([-15.94238515, -8.20712299, -8.22704352, -15.94238515,
                              -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                              -6.52463769, -7.73677889, -15.94238515, -15.94238515,
                              -15.94238515, -15.94238515, -4.18650018, -3.37195286,
                              -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                              -4.70190154, -2.4217066, -15.94238515, -15.94238515,
                              -15.94238515, -15.94238515, -5.62755239, -3.53385194,
                              -15.94238515, -15.94238515, -15.94238515, -15.94238515,
                              -9.43303023, -8.77480925, -15.94238515, -15.94238515,
                              -15.94238515, -15.94238515, -4.2951092, -5.51585994,
                              -15.94238515, -15.94238515, -15.94238515, -4.40151721,
                              -3.95228878, -15.94238515, -15.94238515, -15.94238515,
                              -6.10365415, -4.59494697, -15.94238515, -15.94238515,
                              -15.94238515, -8.10727767, -6.2585298, -15.94238515,
                              -15.94238515, -15.94238515, -5.60161702, -4.47217004,
                              -15.94238515, -15.94238515, -15.94238515, -5.91641988]
                             )
        expected2 = np.array([-15.942385, -8.531508, -8.551396, -15.942385, -15.942385,
                              -15.942385, -15.942385, -15.942385, -5.626043, -6.8381968,
                              -15.942385, -15.942385, -15.942385, -15.942385, -3.3122184,
                              -2.49764, -15.942385, -15.942385, -15.942385, -15.942385,
                              -3.625868, -1.3457257, -15.942385, -15.942385, -15.942385,
                              -15.942385, -4.2223063, -2.1285915, -15.942385, -15.942385,
                              -15.942385, -15.942385, -8.611152, -7.952894, -15.942385,
                              -15.942385, -15.942385, -15.942385, -2.7585578, -3.9793255,
                              -15.942385, -15.942385, -15.942385, -2.5377562, -2.0885658,
                              -15.942385, -15.942385, -15.942385, -3.8310733, -2.322393,
                              -15.942385, -15.942385, -15.942385, -7.674944, -5.8261633,
                              -15.942385, -15.942385, -15.942385, -3.5960004, -2.4665844,
                              -15.942385, -15.942385, -15.942385, -1.7905309]
                             )
        expected3 = np.array([-15.942385, -13.406995, -13.426883, -15.942385, -15.942385,
                              -15.942385, -15.942385, -15.942385, -15.942385, -15.942385,
                              -15.942385, -15.942385, -15.942385, -15.942385, -13.493383,
                              -12.678805, -15.942385, -15.942385, -15.942385, -15.942385,
                              -14.809377, -12.529235, -15.942385, -15.942385, -15.942385,
                              -15.942385, -13.838827, -11.745112, -15.942385, -15.942385,
                              -15.942385, -15.942385, -13.9336405, -13.275384, -15.942385,
                              -15.942385, -15.942385, -15.942385, -13.043786, -14.264554,
                              -15.942385, -15.942385, -15.942385, -13.060181, -12.610991,
                              -15.942385, -15.942385, -15.942385, -14.152064, -12.643384,
                              -15.942385, -15.942385, -15.942385, -14.48317, -12.634389,
                              -15.942385, -15.942385, -15.942385, -14.627316, -13.4979,
                              -15.942385, -15.942385, -15.942385, -12.6279955]
                             )
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:64, 400], expected1, atol=1e-5))
        self.assertTrue(np.allclose(spec_list[1][:64, 400], expected2, atol=1e-5))
        self.assertTrue(np.allclose(spec_list[2][:64, 400], expected3, atol=1e-5))

    def test_spectrogram_center_padding(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="reflect",
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.1287945, 0.12792738, 0.08311573, 0.03155122, 0.02470202,
            0.00727857, 0.00910694, 0.00686163, 0.01238981, 0.01473668,
            0.00336144, 0.00370314, 0.00600871, 0.01120164, 0.01942998,
            0.03132008, 0.0232842, 0.01124642, 0.02754783, 0.02423725,
            0.00147893, 0.00038027, 0.00112299, 0.00596233, 0.00571529,
            0.02084235, 0.0231855, 0.00810006, 0.01837943, 0.00651339,
            0.00093931, 0.00067426, 0.01058399, 0.01270507, 0.00151734,
            0.00331913, 0.00302416, 0.01081792, 0.00754549, 0.00148963,
            0.00111943, 0.00152573, 0.00608017, 0.01749986, 0.01205949,
            0.0143082, 0.01910573, 0.00413786, 0.03916619, 0.09873404,
            0.08302026, 0.02673891, 0.00401255, 0.01397392, 0.00751862,
            0.01024884, 0.01544606, 0.00638907, 0.00623633, 0.0085103,
            0.00217659, 0.00276204, 0.00260835, 0.00299299,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="constant",
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.06558744, 0.06889656, 0.06263352, 0.04264418, 0.03404115,
            0.03244197, 0.02279134, 0.01646339, 0.01452216, 0.00826055,
            0.00062093, 0.0031821, 0.00419456, 0.00689327, 0.01106367,
            0.01712119, 0.01721762, 0.00977533, 0.01606626, 0.02275621,
            0.01727687, 0.00992739, 0.01217688, 0.01049927, 0.01022947,
            0.01302475, 0.01166873, 0.01081812, 0.01057327, 0.00767912,
            0.00429567, 0.00089625, 0.00654583, 0.00912084, 0.00700984,
            0.00225026, 0.00290545, 0.00667712, 0.00730663, 0.00410813,
            0.00073102, 0.00219296, 0.00527618, 0.00996585, 0.01123781,
            0.00872816, 0.01165121, 0.02047945, 0.03681747, 0.0514379,
            0.05137928, 0.03960042, 0.02821562, 0.01813349, 0.01201322,
            0.01260964, 0.00900654, 0.00207905, 0.00456714, 0.00850599,
            0.00788239, 0.00664407, 0.00824227, 0.00628301,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=False,
        )
        self.assertEqual(spec.shape, (257, 728))

        # fmt: off
        expected = np.array([
            0.00250445, 0.02161521, 0.06232229, 0.04339567, 0.00937727,
            0.01080616, 0.00248685, 0.0095264, 0.00727476, 0.0079152,
            0.00839946, 0.00254932, 0.00716622, 0.005559, 0.00272623,
            0.00581774, 0.01896395, 0.01829788, 0.01020514, 0.01632692,
            0.00870888, 0.02065827, 0.0136022, 0.0132382, 0.011827,
            0.00194505, 0.0189979, 0.026874, 0.02194014, 0.01923883,
            0.01621437, 0.00661967, 0.00289517, 0.00470257, 0.00957801,
            0.00191455, 0.00431664, 0.00544359, 0.01126213, 0.00785778,
            0.00423469, 0.01322504, 0.02226548, 0.02318576, 0.03428908,
            0.03648811, 0.0202938, 0.011902, 0.03226198, 0.06347476,
            0.01306318, 0.05308729, 0.05474771, 0.03127991, 0.00998512,
            0.01449977, 0.01272741, 0.00868176, 0.00850386, 0.00313876,
            0.00811857, 0.00538216, 0.00685749, 0.00535275,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

    def test_spectrogram_batch_center_padding(self):
        waveform_list = self._load_datasamples(3)

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="reflect",
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[2].shape, (257, 1561))

        # fmt: off
        expected1 = np.array([
            0.1287945, 0.12792738, 0.08311573, 0.03155122, 0.02470202,
            0.00727857, 0.00910694, 0.00686163, 0.01238981, 0.01473668,
            0.00336144, 0.00370314, 0.00600871, 0.01120164, 0.01942998,
            0.03132008, 0.0232842, 0.01124642, 0.02754783, 0.02423725,
            0.00147893, 0.00038027, 0.00112299, 0.00596233, 0.00571529,
            0.02084235, 0.0231855, 0.00810006, 0.01837943, 0.00651339,
            0.00093931, 0.00067426, 0.01058399, 0.01270507, 0.00151734,
            0.00331913, 0.00302416, 0.01081792, 0.00754549, 0.00148963,
            0.00111943, 0.00152573, 0.00608017, 0.01749986, 0.01205949,
            0.0143082, 0.01910573, 0.00413786, 0.03916619, 0.09873404,
            0.08302026, 0.02673891, 0.00401255, 0.01397392, 0.00751862,
            0.01024884, 0.01544606, 0.00638907, 0.00623633, 0.0085103,
            0.00217659, 0.00276204, 0.00260835, 0.00299299,
        ])
        expected2 = np.array([
            1.89624839e-02, 1.23274978e-02, 3.69160250e-02, 4.76267971e-02,
            1.39258439e-02, 2.98370440e-02, 2.74845166e-03, 3.01934010e-03,
            1.18722776e-02, 9.70834121e-03, 2.06300567e-04, 6.32975250e-04,
            8.20603687e-03, 1.21864351e-02, 3.28791840e-03, 3.36801982e-04,
            2.79373326e-03, 5.00530424e-03, 8.46884679e-03, 1.14089288e-02,
            8.59052036e-03, 2.88538425e-03, 9.95071139e-03, 6.80431770e-03,
            2.95809377e-03, 1.46285209e-04, 3.36268265e-03, 4.80051298e-04,
            2.84506916e-03, 9.34222655e-04, 3.42161348e-03, 2.79612141e-03,
            3.38875921e-03, 2.85030343e-03, 5.39513239e-05, 2.72908504e-03,
            2.09591188e-03, 5.00271388e-04, 8.31917219e-04, 2.37967237e-03,
            1.75001193e-03, 1.31826295e-04, 8.83622793e-04, 1.54303256e-04,
            3.09544569e-03, 4.08527814e-03, 2.73566321e-03, 1.78805250e-03,
            9.53314066e-06, 1.74316950e-03, 1.51099428e-03, 8.65990878e-04,
            8.44859460e-04, 5.35220199e-04, 5.36562002e-04, 8.33181897e-04,
            8.22705682e-04, 1.81083288e-03, 9.75003233e-04, 6.73114730e-04,
            6.81665202e-04, 2.05180887e-03, 1.10151991e-03, 4.75923851e-04,
        ])
        expected3 = np.array([
            0.07079848, 0.04237922, 0.0220724, 0.04446052, 0.03598337,
            0.03327273, 0.02545774, 0.01319528, 0.00919659, 0.01376867,
            0.00361992, 0.00608425, 0.01105873, 0.0105565, 0.00744286,
            0.00244849, 0.00257317, 0.00749989, 0.01061386, 0.01525312,
            0.00656914, 0.01199581, 0.00487319, 0.00830956, 0.0046706,
            0.00588962, 0.00544486, 0.00565179, 0.00050112, 0.01108059,
            0.00217417, 0.00453234, 0.00537306, 0.00269329, 0.00342333,
            0.00095484, 0.00708934, 0.00660373, 0.00543686, 0.00217186,
            0.00431519, 0.00457764, 0.00503529, 0.01166454, 0.01375581,
            0.01467224, 0.00873404, 0.00534086, 0.00476848, 0.0226163,
            0.0314, 0.00151021, 0.01975221, 0.01637519, 0.00046068,
            0.0460544, 0.06285986, 0.03151625, 0.0013598, 0.004804,
            0.0073824, 0.02312599, 0.02613977, 0.01056851
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:64, 0], expected1))
        self.assertTrue(np.allclose(spec_list[1][:64, 0], expected2))
        self.assertTrue(np.allclose(spec_list[2][:64, 0], expected3))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="constant",
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[2].shape, (257, 1561))

        # fmt: off
        expected1 = np.array([
            0.06558744, 0.06889656, 0.06263352, 0.04264418, 0.03404115,
            0.03244197, 0.02279134, 0.01646339, 0.01452216, 0.00826055,
            0.00062093, 0.0031821, 0.00419456, 0.00689327, 0.01106367,
            0.01712119, 0.01721762, 0.00977533, 0.01606626, 0.02275621,
            0.01727687, 0.00992739, 0.01217688, 0.01049927, 0.01022947,
            0.01302475, 0.01166873, 0.01081812, 0.01057327, 0.00767912,
            0.00429567, 0.00089625, 0.00654583, 0.00912084, 0.00700984,
            0.00225026, 0.00290545, 0.00667712, 0.00730663, 0.00410813,
            0.00073102, 0.00219296, 0.00527618, 0.00996585, 0.01123781,
            0.00872816, 0.01165121, 0.02047945, 0.03681747, 0.0514379,
            0.05137928, 0.03960042, 0.02821562, 0.01813349, 0.01201322,
            0.01260964, 0.00900654, 0.00207905, 0.00456714, 0.00850599,
            0.00788239, 0.00664407, 0.00824227, 0.00628301,
        ])
        expected2 = np.array([
            0.00955754, 0.01445548, 0.02393902, 0.02903068, 0.02512844,
            0.01508297, 0.00474784, 0.00440362, 0.0073898, 0.00546519,
            0.00126077, 0.00240507, 0.00523254, 0.00632742, 0.00415215,
            0.00056628, 0.00161288, 0.0026956, 0.00431587, 0.00621471,
            0.00791291, 0.0079454, 0.00594525, 0.00334581, 0.00180047,
            0.00144485, 0.00175764, 0.00188037, 0.00134889, 0.00150253,
            0.00178821, 0.00158875, 0.00204339, 0.00266497, 0.00280556,
            0.00221949, 0.00108956, 0.000532, 0.00108454, 0.00129254,
            0.00089315, 0.00022803, 0.00038176, 0.0011302, 0.00189306,
            0.0021964, 0.00203576, 0.00207306, 0.00217727, 0.00174297,
            0.00103331, 0.00076695, 0.0007422, 0.00061986, 0.00081204,
            0.00079615, 0.00089417, 0.00105452, 0.00042615, 0.00066372,
            0.00132765, 0.00122087, 0.00054903, 0.00107945,
        ])
        expected3 = np.array([
            0.03573493, 0.03625983, 0.03341755, 0.02431477, 0.01770546,
            0.0169356, 0.01579034, 0.01600499, 0.01329064, 0.00747957,
            0.00367372, 0.00403853, 0.00519597, 0.00551022, 0.00532757,
            0.00367569, 0.00130341, 0.00345149, 0.00520744, 0.00872308,
            0.01172503, 0.00948154, 0.00344236, 0.00387997, 0.00425455,
            0.00394357, 0.00711733, 0.00615654, 0.00055756, 0.00656414,
            0.00852001, 0.00666252, 0.00509767, 0.00246784, 0.00376049,
            0.00682879, 0.00641118, 0.00469685, 0.00358701, 0.0015552,
            0.00261458, 0.00701979, 0.00929578, 0.00894536, 0.00828491,
            0.00773528, 0.00552091, 0.00259871, 0.00933179, 0.01588626,
            0.01697887, 0.01268552, 0.00957255, 0.01204092, 0.02123362,
            0.03062669, 0.03215763, 0.02629963, 0.01769568, 0.01088869,
            0.01151334, 0.01378197, 0.01319263, 0.01066859,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:64, 0], expected1))
        self.assertTrue(np.allclose(spec_list[1][:64, 0], expected2))
        self.assertTrue(np.allclose(spec_list[2][:64, 0], expected3))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=False,
        )
        self.assertEqual(spec_list[0].shape, (257, 728))
        self.assertEqual(spec_list[1].shape, (257, 598))
        self.assertEqual(spec_list[2].shape, (257, 1557))

        # fmt: off
        expected1 = np.array([
            0.00250445, 0.02161521, 0.06232229, 0.04339567, 0.00937727,
            0.01080616, 0.00248685, 0.0095264, 0.00727476, 0.0079152,
            0.00839946, 0.00254932, 0.00716622, 0.005559, 0.00272623,
            0.00581774, 0.01896395, 0.01829788, 0.01020514, 0.01632692,
            0.00870888, 0.02065827, 0.0136022, 0.0132382, 0.011827,
            0.00194505, 0.0189979, 0.026874, 0.02194014, 0.01923883,
            0.01621437, 0.00661967, 0.00289517, 0.00470257, 0.00957801,
            0.00191455, 0.00431664, 0.00544359, 0.01126213, 0.00785778,
            0.00423469, 0.01322504, 0.02226548, 0.02318576, 0.03428908,
            0.03648811, 0.0202938, 0.011902, 0.03226198, 0.06347476,
            0.01306318, 0.05308729, 0.05474771, 0.03127991, 0.00998512,
            0.01449977, 0.01272741, 0.00868176, 0.00850386, 0.00313876,
            0.00811857, 0.00538216, 0.00685749, 0.00535275,
        ])
        expected2 = np.array([
            0.01232908, 0.05980514, 0.08285419, 0.01850723, 0.02823627,
            0.00204369, 0.01372626, 0.00956435, 0.02267217, 0.00947112,
            0.00355174, 0.00418008, 0.00843608, 0.01559252, 0.01125505,
            0.00183573, 0.00765051, 0.0109983, 0.00890545, 0.00583453,
            0.00115901, 0.00579039, 0.00151353, 0.00395812, 0.00231413,
            0.00384272, 0.00313914, 0.00072331, 0.00338935, 0.00383328,
            0.00218129, 0.00284516, 0.00228538, 0.00083603, 0.00111663,
            0.00235799, 0.00142748, 0.00092908, 0.0012966, 0.0011403,
            0.0010619, 0.00158732, 0.00289866, 0.00216709, 0.00313325,
            0.00361277, 0.00202507, 0.0009948, 0.00114428, 0.00200851,
            0.0009234, 0.00063468, 0.00018746, 0.00100463, 0.00053799,
            0.00080009, 0.00158291, 0.00172077, 0.00173586, 0.00197127,
            0.00107058, 0.00043486, 0.0009859, 0.00215484,
        ])
        expected3 = np.array([
            0.01864123, 0.06131337, 0.08346292, 0.04936386, 0.02792609,
            0.01005205, 0.00884826, 0.02198604, 0.02421535, 0.00957573,
            0.00503561, 0.00241331, 0.00175652, 0.00195889, 0.00453299,
            0.0020317, 0.00249264, 0.00517483, 0.01111943, 0.0150079,
            0.01977743, 0.01253825, 0.00517561, 0.01031712, 0.00579466,
            0.00783679, 0.0071415, 0.00591847, 0.01510728, 0.01194921,
            0.00518072, 0.00125978, 0.00577552, 0.01050614, 0.0077644,
            0.0042905, 0.00278469, 0.00166695, 0.00255013, 0.00578153,
            0.00586451, 0.00929514, 0.01501226, 0.00741419, 0.00310625,
            0.00086757, 0.00595618, 0.0053882, 0.0116266, 0.02504773,
            0.02889692, 0.03739442, 0.04730207, 0.03856638, 0.05700104,
            0.04299267, 0.02153366, 0.03740607, 0.03811468, 0.01575022,
            0.00676344, 0.01359865, 0.01769319, 0.00907966,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:64, 0], expected1))
        self.assertTrue(np.allclose(spec_list[1][:64, 0], expected2))
        self.assertTrue(np.allclose(spec_list[2][:64, 0], expected3))

    def test_spectrogram_shapes(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (201, 732))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (201, 729))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 1464))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 1464))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 183))

    def test_spectrogram_batch_shapes(self):
        waveform_list = self._load_datasamples(3)

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec_list[0].shape, (201, 732))
        self.assertEqual(spec_list[1].shape, (201, 602))
        self.assertEqual(spec_list[2].shape, (201, 1561))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec_list[0].shape, (201, 729))
        self.assertEqual(spec_list[1].shape, (201, 599))
        self.assertEqual(spec_list[2].shape, (201, 1558))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[2].shape, (257, 1561))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec_list[0].shape, (512, 1464))
        self.assertEqual(spec_list[1].shape, (512, 1204))
        self.assertEqual(spec_list[2].shape, (512, 3122))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec_list[0].shape, (512, 1464))
        self.assertEqual(spec_list[1].shape, (512, 1204))
        self.assertEqual(spec_list[2].shape, (512, 3122))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec_list[0].shape, (512, 183))
        self.assertEqual(spec_list[1].shape, (512, 151))
        self.assertEqual(spec_list[2].shape, (512, 391))

    def test_mel_spectrogram(self):
        waveform = self._load_datasamples(1)[0]

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm=None,
            mel_scale="htk",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

        spec = spectrogram(
            waveform,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec.shape, (513, 732))

        spec = spectrogram(
            waveform,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
            mel_filters=mel_filters,
        )
        self.assertEqual(spec.shape, (13, 732))

        # fmt: off
        expected = np.array([
            1.08027889e+02, 1.48080673e+01, 7.70758213e+00, 9.57676639e-01,
            8.81639061e-02, 5.26073833e-02, 1.52736155e-02, 9.95350117e-03,
            7.95364356e-03, 1.01148004e-02, 4.29241020e-03, 9.90708797e-03,
            9.44153646e-04
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:, 300], expected))

    def test_mel_spectrogram_batch(self):
        waveform_list = self._load_datasamples(3)

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm=None,
            mel_scale="htk",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec_list[0].shape, (513, 732))
        self.assertEqual(spec_list[1].shape, (513, 602))
        self.assertEqual(spec_list[2].shape, (513, 1561))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
            mel_filters=mel_filters,
        )
        self.assertEqual(spec_list[0].shape, (13, 732))
        self.assertEqual(spec_list[1].shape, (13, 602))
        self.assertEqual(spec_list[2].shape, (13, 1561))

        # fmt: off
        expected1 = np.array([
            1.08027889e+02, 1.48080673e+01, 7.70758213e+00, 9.57676639e-01,
            8.81639061e-02, 5.26073833e-02, 1.52736155e-02, 9.95350117e-03,
            7.95364356e-03, 1.01148004e-02, 4.29241020e-03, 9.90708797e-03,
            9.44153646e-04
        ])
        expected2 = np.array([
            71.82577165, 109.44693334, 272.4834194, 164.90450355,
            16.54056349, 11.60810547, 24.87525946, 21.07317022,
            1.26736284, 1.4583074, 1.36659061, 1.76305768,
            2.03703503
        ])
        expected3 = np.array([
            5.22246749e+02, 6.92660728e+02, 2.65895922e+02, 2.06526565e+01,
            2.28692104e+00, 1.19473622e+00, 8.43228216e-01, 3.20760592e+00,
            1.33654151e+00, 1.51050684e-01, 2.78282477e-01, 9.25020981e-01,
            2.29908841e-01
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][:, 300], expected1))
        self.assertTrue(np.allclose(spec_list[1][:, 300], expected2))
        self.assertTrue(np.allclose(spec_list[2][:, 300], expected3))

    def test_spectrogram_power(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=None,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.complex64)

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.float64)

        # fmt: off
        expected = np.array([
            0.02328461, 0.02390484, 0.01978448, 0.04115711, 0.0624309,
            0.05197181, 0.05896072, 0.08839577, 0.07726794, 0.06432579,
            0.11063128, 0.13762532, 0.10935163, 0.11911998, 0.15112405,
            0.14588428, 0.18860507, 0.23992978, 0.15910825, 0.04793241,
            0.07462307, 0.10001811, 0.06125769, 0.05411011, 0.10342509,
            0.09549777, 0.05892122, 0.06534349, 0.06569936, 0.05870678,
            0.10856833, 0.1524107, 0.11463385, 0.05766969, 0.12385171,
            0.14472842, 0.11978184, 0.10353675, 0.07244056, 0.03461861,
            0.02624896, 0.02227475, 0.01238363, 0.00885281, 0.0110049,
            0.00807005, 0.01033663, 0.01703181, 0.01445856, 0.00585615,
            0.0132431, 0.02754132, 0.01524478, 0.0204908, 0.07453328,
            0.10716327, 0.07195779, 0.08816078, 0.18340898, 0.16449876,
            0.12322842, 0.1621659, 0.12334293, 0.06033659,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[64:128, 321], expected))

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.float64)

        # fmt: off
        expected = np.array([
            5.42173162e-04, 5.71441371e-04, 3.91425507e-04, 1.69390778e-03,
            3.89761780e-03, 2.70106923e-03, 3.47636663e-03, 7.81381316e-03,
            5.97033510e-03, 4.13780799e-03, 1.22392802e-02, 1.89407300e-02,
            1.19577805e-02, 1.41895693e-02, 2.28384770e-02, 2.12822221e-02,
            3.55718732e-02, 5.75663000e-02, 2.53154356e-02, 2.29751552e-03,
            5.56860259e-03, 1.00036217e-02, 3.75250424e-03, 2.92790355e-03,
            1.06967501e-02, 9.11982451e-03, 3.47171025e-03, 4.26977174e-03,
            4.31640586e-03, 3.44648538e-03, 1.17870830e-02, 2.32290216e-02,
            1.31409196e-02, 3.32579296e-03, 1.53392460e-02, 2.09463164e-02,
            1.43476883e-02, 1.07198600e-02, 5.24763530e-03, 1.19844836e-03,
            6.89007982e-04, 4.96164430e-04, 1.53354369e-04, 7.83722571e-05,
            1.21107812e-04, 6.51257360e-05, 1.06845939e-04, 2.90082477e-04,
            2.09049831e-04, 3.42945241e-05, 1.75379610e-04, 7.58524227e-04,
            2.32403356e-04, 4.19872697e-04, 5.55520924e-03, 1.14839673e-02,
            5.17792348e-03, 7.77232368e-03, 3.36388536e-02, 2.70598419e-02,
            1.51852425e-02, 2.62977779e-02, 1.52134784e-02, 3.64050455e-03,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[64:128, 321], expected))

    def test_spectrogram_batch_power(self):
        waveform_list = self._load_datasamples(3)

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=None,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[0].dtype, np.complex64)
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[1].dtype, np.complex64)
        self.assertEqual(spec_list[2].shape, (257, 1561))
        self.assertEqual(spec_list[2].dtype, np.complex64)

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[0].dtype, np.float64)
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[1].dtype, np.float64)
        self.assertEqual(spec_list[2].shape, (257, 1561))
        self.assertEqual(spec_list[2].dtype, np.float64)

        # fmt: off
        expected1 = np.array([
            0.02328461, 0.02390484, 0.01978448, 0.04115711, 0.0624309,
            0.05197181, 0.05896072, 0.08839577, 0.07726794, 0.06432579,
            0.11063128, 0.13762532, 0.10935163, 0.11911998, 0.15112405,
            0.14588428, 0.18860507, 0.23992978, 0.15910825, 0.04793241,
            0.07462307, 0.10001811, 0.06125769, 0.05411011, 0.10342509,
            0.09549777, 0.05892122, 0.06534349, 0.06569936, 0.05870678,
            0.10856833, 0.1524107, 0.11463385, 0.05766969, 0.12385171,
            0.14472842, 0.11978184, 0.10353675, 0.07244056, 0.03461861,
            0.02624896, 0.02227475, 0.01238363, 0.00885281, 0.0110049,
            0.00807005, 0.01033663, 0.01703181, 0.01445856, 0.00585615,
            0.0132431, 0.02754132, 0.01524478, 0.0204908, 0.07453328,
            0.10716327, 0.07195779, 0.08816078, 0.18340898, 0.16449876,
            0.12322842, 0.1621659, 0.12334293, 0.06033659,
        ])
        expected2 = np.array([
            0.01778026, 0.00929138, 0.00692273, 0.00927352, 0.01261294,
            0.01237128, 0.00852516, 0.00171938, 0.00727061, 0.00716808,
            0.00909281, 0.01289532, 0.01469949, 0.01499858, 0.01332855,
            0.02296907, 0.01706539, 0.00773101, 0.01666623, 0.02311021,
            0.0413901, 0.07787261, 0.10634092, 0.09296556, 0.05218428,
            0.01813716, 0.00546139, 0.01470388, 0.02515159, 0.0192187,
            0.01222719, 0.00744678, 0.01045674, 0.01923522, 0.01990819,
            0.01174323, 0.01535391, 0.02786647, 0.02904595, 0.0313408,
            0.0340503, 0.03118268, 0.02915136, 0.04200513, 0.05563153,
            0.05429446, 0.05021769, 0.05882667, 0.06668596, 0.06555867,
            0.04523559, 0.01489498, 0.01031892, 0.02134155, 0.01736669,
            0.0195216, 0.03971575, 0.03938636, 0.02052712, 0.03104931,
            0.0902727, 0.09022622, 0.03275532, 0.0172633,
        ])
        expected3 = np.array([
            0.04684551, 0.08238806, 0.05658358, 0.01653778, 0.06498249,
            0.09553589, 0.10281084, 0.09191031, 0.07000408, 0.06737158,
            0.06534155, 0.06675509, 0.09008541, 0.10184046, 0.09783596,
            0.0963737, 0.08520112, 0.05370093, 0.03453015, 0.03648568,
            0.06339967, 0.09340346, 0.09417402, 0.08623119, 0.07175977,
            0.04406138, 0.04796988, 0.05407591, 0.0471824, 0.04022626,
            0.06438748, 0.0808218, 0.0745263, 0.06191467, 0.03116328,
            0.03206497, 0.05867718, 0.04424652, 0.04448404, 0.07032498,
            0.08300796, 0.07895744, 0.0816894, 0.09392357, 0.07571699,
            0.03967651, 0.07703795, 0.06464871, 0.08704693, 0.14085226,
            0.1350321, 0.18794712, 0.27043005, 0.26596246, 0.19948336,
            0.06545141, 0.13204652, 0.08554521, 0.2262849, 0.33900721,
            0.3970475, 0.3482436, 0.17134947, 0.46249565,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][64:128, 321], expected1))
        self.assertTrue(np.allclose(spec_list[1][64:128, 321], expected2))
        self.assertTrue(np.allclose(spec_list[2][64:128, 321], expected3))

        spec_list = spectrogram_batch(
            waveform_list,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec_list[0].shape, (257, 732))
        self.assertEqual(spec_list[0].dtype, np.float64)
        self.assertEqual(spec_list[1].shape, (257, 602))
        self.assertEqual(spec_list[1].dtype, np.float64)
        self.assertEqual(spec_list[2].shape, (257, 1561))
        self.assertEqual(spec_list[2].dtype, np.float64)

        # fmt: off
        expected1 = np.array([
            5.42173162e-04, 5.71441371e-04, 3.91425507e-04, 1.69390778e-03,
            3.89761780e-03, 2.70106923e-03, 3.47636663e-03, 7.81381316e-03,
            5.97033510e-03, 4.13780799e-03, 1.22392802e-02, 1.89407300e-02,
            1.19577805e-02, 1.41895693e-02, 2.28384770e-02, 2.12822221e-02,
            3.55718732e-02, 5.75663000e-02, 2.53154356e-02, 2.29751552e-03,
            5.56860259e-03, 1.00036217e-02, 3.75250424e-03, 2.92790355e-03,
            1.06967501e-02, 9.11982451e-03, 3.47171025e-03, 4.26977174e-03,
            4.31640586e-03, 3.44648538e-03, 1.17870830e-02, 2.32290216e-02,
            1.31409196e-02, 3.32579296e-03, 1.53392460e-02, 2.09463164e-02,
            1.43476883e-02, 1.07198600e-02, 5.24763530e-03, 1.19844836e-03,
            6.89007982e-04, 4.96164430e-04, 1.53354369e-04, 7.83722571e-05,
            1.21107812e-04, 6.51257360e-05, 1.06845939e-04, 2.90082477e-04,
            2.09049831e-04, 3.42945241e-05, 1.75379610e-04, 7.58524227e-04,
            2.32403356e-04, 4.19872697e-04, 5.55520924e-03, 1.14839673e-02,
            5.17792348e-03, 7.77232368e-03, 3.36388536e-02, 2.70598419e-02,
            1.51852425e-02, 2.62977779e-02, 1.52134784e-02, 3.64050455e-03,
        ])
        expected2 = np.array([
            3.16137604e-04, 8.63297362e-05, 4.79241720e-05, 8.59982493e-05,
            1.59086326e-04, 1.53048476e-04, 7.26783945e-05, 2.95627100e-06,
            5.28617352e-05, 5.13813355e-05, 8.26792588e-05, 1.66289156e-04,
            2.16075069e-04, 2.24957314e-04, 1.77650211e-04, 5.27578282e-04,
            2.91227688e-04, 5.97685493e-05, 2.77763360e-04, 5.34081651e-04,
            1.71314057e-03, 6.06414277e-03, 1.13083916e-02, 8.64259617e-03,
            2.72319867e-03, 3.28956593e-04, 2.98268126e-05, 2.16204145e-04,
            6.32602626e-04, 3.69358508e-04, 1.49504171e-04, 5.54544917e-05,
            1.09343371e-04, 3.69993847e-04, 3.96335839e-04, 1.37903521e-04,
            2.35742483e-04, 7.76540114e-04, 8.43667068e-04, 9.82245923e-04,
            1.15942286e-03, 9.72359636e-04, 8.49801853e-04, 1.76443092e-03,
            3.09486753e-03, 2.94788822e-03, 2.52181630e-03, 3.46057723e-03,
            4.44701769e-03, 4.29793858e-03, 2.04625858e-03, 2.21860290e-04,
            1.06480179e-04, 4.55461892e-04, 3.01601836e-04, 3.81092892e-04,
            1.57734053e-03, 1.55128531e-03, 4.21362677e-04, 9.64059883e-04,
            8.14916019e-03, 8.14077014e-03, 1.07291131e-03, 2.98021545e-04,
        ])
        expected3 = np.array([
            0.0021945, 0.00678779, 0.0032017, 0.0002735, 0.00422272,
            0.00912711, 0.01057007, 0.00844751, 0.00490057, 0.00453893,
            0.00426952, 0.00445624, 0.00811538, 0.01037148, 0.00957188,
            0.00928789, 0.00725923, 0.00288379, 0.00119233, 0.0013312,
            0.00401952, 0.00872421, 0.00886875, 0.00743582, 0.00514946,
            0.00194141, 0.00230111, 0.0029242, 0.00222618, 0.00161815,
            0.00414575, 0.00653216, 0.00555417, 0.00383343, 0.00097115,
            0.00102816, 0.00344301, 0.00195775, 0.00197883, 0.0049456,
            0.00689032, 0.00623428, 0.00667316, 0.00882164, 0.00573306,
            0.00157423, 0.00593485, 0.00417946, 0.00757717, 0.01983936,
            0.01823367, 0.03532412, 0.07313241, 0.07073603, 0.03979361,
            0.00428389, 0.01743628, 0.00731798, 0.05120486, 0.11492589,
            0.15764671, 0.1212736, 0.02936064, 0.21390222
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec_list[0][64:128, 321], expected1))
        self.assertTrue(np.allclose(spec_list[1][64:128, 321], expected2))
        self.assertTrue(np.allclose(spec_list[2][64:128, 321], expected3))

    def test_power_to_db(self):
        tmp_spectrogram = np.zeros((2, 3))
        tmp_spectrogram[0, 0] = 2.0
        tmp_spectrogram[0, 1] = 0.5
        tmp_spectrogram[0, 2] = 0.707
        tmp_spectrogram[1, 1] = 1.0

        output = power_to_db(tmp_spectrogram, reference=1.0)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(tmp_spectrogram, reference=2.0)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-103.01029996, -3.01029996, -103.01029996]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(tmp_spectrogram, min_value=1e-6)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(tmp_spectrogram, db_range=80)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-76.98970004, 0.0, -76.98970004]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(tmp_spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-80.0, -3.01029996, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(tmp_spectrogram, reference=2.0, min_value=1e-6, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-63.01029996, -3.01029996, -63.01029996]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            power_to_db(tmp_spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            power_to_db(tmp_spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            power_to_db(tmp_spectrogram, db_range=-80)

    def test_power_to_db_batch(self):
        # Setup a batch of spectrograms with varying values and lengths
        batch_spectrogram = np.zeros((3, 2, 3))
        batch_spectrogram[0, 0, 0] = 2.0
        batch_spectrogram[0, 0, 1] = 0.5
        batch_spectrogram[0, 0, 2] = 0.707
        batch_spectrogram[0, 1, 1] = 1.0
        batch_spectrogram[1, :, :2] = batch_spectrogram[0, :, :2] * 1.5
        batch_spectrogram[2, :, :1] = batch_spectrogram[0, :, :1] * 0.5

        # Expected values computed by applying `power_to_db` iteratively
        output = power_to_db_batch(batch_spectrogram, reference=1.0)
        expected = np.array(
            [
                [[3.01029996, -3.01029996, -1.50580586], [-100, 0, -100]],
                [[4.77121255, -1.24938737, -100], [-100, 1.76091259, -100]],
                [[0, -100, -100], [-100, -100, -100]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db_batch(batch_spectrogram, reference=2.0)
        expected = np.array(
            [
                [[0, -6.02059991, -4.51610582], [-103.01029996, -3.01029996, -103.01029996]],
                [[1.76091259, -4.25968732, -103.01029996], [-103.01029996, -1.24938737, -103.01029996]],
                [[-3.01029996, -103.01029996, -103.01029996], [-103.01029996, -103.01029996, -103.01029996]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db_batch(batch_spectrogram, min_value=1e-6)
        expected = np.array(
            [
                [[3.01029996, -3.01029996, -1.50580586], [-60, 0, -60]],
                [[4.77121255, -1.24938737, -60], [-60, 1.76091259, -60]],
                [[0, -60, -60], [-60, -60, -60]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db_batch(batch_spectrogram, db_range=80)
        expected = np.array(
            [
                [[3.01029996, -3.01029996, -1.50580586], [-76.98970004, 0, -76.98970004]],
                [[4.77121255, -1.24938737, -75.22878745], [-75.22878745, 1.76091259, -75.22878745]],
                [[0, -80, -80], [-80, -80, -80]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db_batch(batch_spectrogram, reference=2.0, db_range=80)
        expected = np.array(
            [
                [[0, -6.02059991, -4.51610582], [-80, -3.01029996, -80]],
                [[1.76091259, -4.25968732, -78.23908741], [-78.23908741, -1.24938737, -78.23908741]],
                [[-3.01029996, -83.01029996, -83.01029996], [-83.01029996, -83.01029996, -83.01029996]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db_batch(batch_spectrogram, reference=2.0, min_value=1e-6, db_range=80)
        expected = np.array(
            [
                [[0, -6.02059991, -4.51610582], [-63.01029996, -3.01029996, -63.01029996]],
                [[1.76091259, -4.25968732, -63.01029996], [-63.01029996, -1.24938737, -63.01029996]],
                [[-3.01029996, -63.01029996, -63.01029996], [-63.01029996, -63.01029996, -63.01029996]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            power_to_db_batch(batch_spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            power_to_db_batch(batch_spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            power_to_db_batch(batch_spectrogram, db_range=-80)

    def test_amplitude_to_db(self):
        tmp_spectrogram = np.zeros((2, 3))
        tmp_spectrogram[0, 0] = 2.0
        tmp_spectrogram[0, 1] = 0.5
        tmp_spectrogram[0, 2] = 0.707
        tmp_spectrogram[1, 1] = 1.0

        output = amplitude_to_db(tmp_spectrogram, reference=1.0)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(tmp_spectrogram, reference=2.0)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-106.02059991, -6.02059991, -106.02059991]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(tmp_spectrogram, min_value=1e-3)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(tmp_spectrogram, db_range=80)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-73.97940009, 0.0, -73.97940009]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(tmp_spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-80.0, -6.02059991, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(tmp_spectrogram, reference=2.0, min_value=1e-3, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-66.02059991, -6.02059991, -66.02059991]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            amplitude_to_db(tmp_spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(tmp_spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(tmp_spectrogram, db_range=-80)

    def test_amplitude_to_db_batch(self):
        # Setup a batch of spectrograms with varying values and lengths
        batch_spectrogram = np.zeros((3, 2, 3))
        batch_spectrogram[0, 0, 0] = 2.0
        batch_spectrogram[0, 0, 1] = 0.5
        batch_spectrogram[0, 0, 2] = 0.707
        batch_spectrogram[0, 1, 1] = 1.0
        batch_spectrogram[1, :, :2] = batch_spectrogram[0, :, :2] * 1.5
        batch_spectrogram[2, :, :1] = batch_spectrogram[0, :, :1] * 0.5

        # Expected values computed by applying `amplitude_to_db` iteratively
        output = amplitude_to_db_batch(batch_spectrogram, reference=1.0)
        expected = np.array(
            [
                [[6.02059991, -6.02059991, -3.01161172], [-100, 0, -100]],
                [[9.54242509, -2.49877473, -100], [-100, 3.52182518, -100]],
                [[0, -100, -100], [-100, -100, -100]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db_batch(batch_spectrogram, reference=2.0)
        expected = np.array(
            [
                [[0, -12.04119983, -9.03221164], [-106.02059991, -6.02059991, -106.02059991]],
                [[3.52182518, -8.51937465, -106.02059991], [-106.02059991, -2.49877473, -106.02059991]],
                [[-6.02059991, -106.02059991, -106.02059991], [-106.02059991, -106.02059991, -106.02059991]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db_batch(batch_spectrogram, min_value=1e-3)
        expected = np.array(
            [
                [[6.02059991, -6.02059991, -3.01161172], [-60, 0, -60]],
                [[9.54242509, -2.49877473, -60], [-60, 3.52182518, -60]],
                [[0, -60, -60], [-60, -60, -60]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db_batch(batch_spectrogram, db_range=80)
        expected = np.array(
            [
                [[6.02059991, -6.02059991, -3.01161172], [-73.97940009, 0, -73.97940009]],
                [[9.54242509, -2.49877473, -70.45757491], [-70.45757491, 3.52182518, -70.45757491]],
                [[0, -80, -80], [-80, -80, -80]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db_batch(batch_spectrogram, reference=2.0, db_range=80)
        expected = np.array(
            [
                [[0, -12.04119983, -9.03221164], [-80, -6.02059991, -80]],
                [[3.52182518, -8.51937465, -76.47817482], [-76.47817482, -2.49877473, -76.47817482]],
                [[-6.02059991, -86.02059991, -86.02059991], [-86.02059991, -86.02059991, -86.02059991]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db_batch(batch_spectrogram, reference=2.0, min_value=1e-3, db_range=80)
        expected = np.array(
            [
                [[0, -12.04119983, -9.03221164], [-66.02059991, -6.02059991, -66.02059991]],
                [[3.52182518, -8.51937465, -66.02059991], [-66.02059991, -2.49877473, -66.02059991]],
                [[-6.02059991, -66.02059991, -66.02059991], [-66.02059991, -66.02059991, -66.02059991]],
            ]
        )
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            amplitude_to_db_batch(batch_spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db_batch(batch_spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db_batch(batch_spectrogram, db_range=-80)

    @require_librosa
    def test_chroma_equivalence(self):
        num_frequency_bins = 25
        num_chroma = 6
        sampling_rate = 24000

        # test default parameters
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins, num_chroma=num_chroma, sampling_rate=sampling_rate
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test no weighting_parameters
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins, octwidth=None)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins,
            num_chroma=num_chroma,
            sampling_rate=sampling_rate,
            weighting_parameters=None,
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test with L1 norm
        original_chroma = chroma(sr=sampling_rate, n_chroma=num_chroma, n_fft=num_frequency_bins, norm=1.0)
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins, num_chroma=num_chroma, sampling_rate=sampling_rate, power=1.0
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

        # test starting at 'A' chroma, power = None, tuning = 0, different weighting_parameters
        original_chroma = chroma(
            sr=sampling_rate,
            n_chroma=num_chroma,
            n_fft=num_frequency_bins,
            norm=None,
            base_c=None,
            octwidth=1.0,
            ctroct=4.0,
        )
        utils_chroma = chroma_filter_bank(
            num_frequency_bins=num_frequency_bins,
            num_chroma=num_chroma,
            sampling_rate=sampling_rate,
            power=None,
            start_at_c_chroma=False,
            weighting_parameters=(4.0, 1.0),
        )

        self.assertTrue(np.allclose(original_chroma, utils_chroma))

    def test_stft(self):
        "New added for mindformers' DT coverage."
        np.random.seed(0)
        audio = np.random.rand(50)
        fft_window_size = 10
        hop_length = 2
        framed_audio = fram_wave(audio, hop_length, fft_window_size)
        tmp_spectrogram = stft(framed_audio, None)
        assert tmp_spectrogram.shape == (6, 26)
        assert np.allclose(tmp_spectrogram[0], np.array(
            [5.666678, 5.7816215, 6.489411, 6.400574, 6.179561, 6.207656, 5.231522, 4.986863, 5.660327, 5.8240256,
             5.7261963, 6.402123, 6.2578635, 5.0884504, 5.0581965, 4.746617, 5.1930737, 5.2872386, 5.649349, 5.5536284,
             5.693397, 5.344686, 4.2283587, 3.8661695, 3.0708153, 3.2248435
             ]), rtol=1.e-3, atol=1.e-3)
