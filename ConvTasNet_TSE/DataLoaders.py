import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from AudioReader import AudioReader
import torch.nn.functional as F
import random
import pandas as pd



def TranscriptReader(text_path):
    """
    处理CSV文件，生成音频名字到合并文本的字典
    Args: text_path (str): CSV文件路径
    Returns: dict: 字典，键为音频名字，值为合并后的文本
    """
    try:
        # 读取CSV文件，有表头
        df = pd.read_csv(text_path, header=0, encoding='utf-8')
        # 创建结果字典
        text_dict = {}
        # 遍历每一行数据
        for index, row in df.iterrows():
            # 使用表头对应的列名
            audio_name = str(row['ID']).strip()  # 音频名字（ID列）
            speaker_text = str(row['Speaker']).strip() if pd.notna(row['Speaker']) else ""  # 第一个说话人文本
            text_dict[audio_name] = speaker_text
        print(f"成功处理 {len(text_dict)} 条音频文本数据")
        return text_dict
        
    except FileNotFoundError:
        print(f"错误：文件 {text_path} 不存在")
        return {}
    except KeyError as e:
        print(f"错误：CSV文件中缺少必要的列 {e}")
        print("请确保CSV文件包含以下列：ID, Speaker")
        return {}
    except Exception as e:
        print(f"处理文件时出错：{e}")
        return {}


def pad_collate(batch):
    """
    batch: list of dict
    """
    mixes = [b["mix"] for b in batch]
    targets = [b["target"] for b in batch]

    mix_lens = torch.tensor([m.shape[-1] for m in mixes])
    target_lens = torch.tensor([t.shape[-1] for t in targets])

    max_len = max(mix_lens)

    def pad_1d(x, max_len):
        return F.pad(x, (0, max_len - x.shape[-1]))

    mixes = torch.stack([pad_1d(m, max_len) for m in mixes])
    targets = torch.stack([pad_1d(t, max_len) for t in targets])

    transcripts = [b["transcript"] for b in batch]
    target_spk = [b["target_spk"] for b in batch]

    return {
        "mix": mixes,
        "target": targets,
        "mix_len": mix_lens,
        "transcript": transcripts,
        "target_spk": target_spk
    }

   

def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    # chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoader(dataset,
                    num_workers=num_workers,
                    batch_size=batch_size ,
                    shuffle=is_train,
                    collate_fn=pad_collate)


class Datasets(Dataset):
    def __init__(self, mix_scp, tar_scp_spk1, tar_scp_spk2, txt_scp_spk1, txt_scp_spk2, sr=8000):
        super().__init__()

        self.mix_audio = AudioReader(mix_scp, sample_rate=sr)
        self.target_audio = {
            "spk1":AudioReader(tar_scp_spk1, sample_rate=sr),
            "spk2":AudioReader(tar_scp_spk2, sample_rate=sr)
        }
        self.transcripts = {
            "spk1":TranscriptReader(txt_scp_spk1),
            "spk2":TranscriptReader(txt_scp_spk2)
        }

        self.samples = []

        for mix_id in self.mix_audio.keys:
            # libri2mix 每个 mix 固定有 spk1 / spk2
            self.samples.append((mix_id, "spk1"))
            self.samples.append((mix_id, "spk2"))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        mix_id, spk = self.samples[index]

        mix = self.mix_audio[mix_id]

        target = self.target_audio[spk][mix_id]
        transcript = self.transcripts[spk].get(mix_id, "")

        return {
            "mix": mix,
            "target": target,
            "transcript": transcript,
            "target_spk": spk
        }



if __name__ == "__main__":
    datasets = Datasets('/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/data_scp/data_clean100_scp/cv_mix.scp',
                        '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/data_scp/data_clean100_scp/cv_s1.scp',
                        '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/data_scp/data_clean100_scp/cv_s2.scp',
                        '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/val/val_spk1.csv',
                        '/home/student/zt/Conv-TasNet_/Conv_TasNet_TSE/data/text/val/val_spk2.csv')
    dataloaders = DataLoader(datasets, num_workers=0,
                              batch_size=10, shuffle=False, collate_fn=pad_collate)
    for eg in dataloaders:
        print(eg)
        import pdb
        pdb.set_trace()
