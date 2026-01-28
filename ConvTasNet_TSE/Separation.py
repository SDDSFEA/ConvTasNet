import os
import torch
import sys
sys.path.append('./config')
from AudioReader import AudioReader, write_wav
import argparse
from torch.nn.parallel import data_parallel
from Conv_TasNet_FiLM import ConvTasNet
from utils import get_logger
from DataLoaders import TranscriptReader
from option import parse
import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

class Separation():
    def __init__(self, mix_scp, txt_scp_spk1, txt_scp_spk2, yaml_path, model, gpuid):
        super(Separation, self).__init__()
        # self.mix = AudioReader(mix_path, sample_rate=16000)
        self.mix_audio = AudioReader(mix_scp, sample_rate=8000)
        self.transcripts = {
            "spk1":TranscriptReader(txt_scp_spk1),
            "spk2":TranscriptReader(txt_scp_spk2)
        }
        self.samples = []
        for mix_id in self.mix_audio.keys:
            # libri2mix 每个 mix 固定有 spk1 / spk2
            self.samples.append((mix_id, "spk1"))
            self.samples.append((mix_id, "spk2"))
        # self.text = handle_text(text_scp)
        opt = parse(yaml_path, is_tain=False)
        # os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpuid[0]}"
        print("gpuid:",gpuid)   
        net = ConvTasNet(**opt['net_conf'])
        dicts = torch.load(model, map_location='cpu')
        net.load_state_dict(dicts["model_state_dict"])
        self.logger = get_logger(__name__)
        self.logger.info('Load checkpoint from {}, epoch {: d}'.format(model, dicts["epoch"]))
        self.net=net.cuda()
        self.device=torch.device('cuda:{}'.format(
            gpuid[0]) if len(gpuid) > 0 else 'cpu')
        print("self.device:",self.device)
        self.gpuid=tuple(gpuid)

    def inference(self, file_path):
        with torch.no_grad():
            for index, (mix_id, spk) in tqdm.tqdm(enumerate(self.samples),total=len(self.samples)):
                #self.logger.info("Compute on utterance {}...".format(key))
                # egs=egs.to(self.device)
                mix = self.mix_audio[mix_id]
                mix = mix.to(self.device)  # self.device 应该是 GPU
                # target = self.target_audio[spk][mix_id]
                transcript = self.transcripts[spk].get(mix_id, "")
                # if key not in self.text:
                #     print("key not in self.text:",key)
                # text = self.text[key]
                # text = self.text.get(key, "")
                
                norm = torch.norm(mix,float('inf'))
                # if len(self.gpuid) != 0:
                    # ests=self.net(mix,transcript)
                    # spks=[torch.squeeze(s.detach().cpu()) for s in ests]
                # else:
                est=self.net(mix,transcript)
                    # spks=[torch.squeeze(s.detach()) for s in ests]
                # index=0
                # for s in spks:
                # 裁剪到输入长度
                est = est[:mix.shape[0]]
                # 归一化到输入范围
                est = est * norm / torch.max(torch.abs(est))
                # 转成 [1, T] 保存
                est = est.unsqueeze(0).cpu()
                # 保存
                save_dir = os.path.join(file_path, spk)
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, mix_id)
                write_wav(filename, est, 8000)
            self.logger.info("Compute over {:d} utterances".format(len(self.mix_audio)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='/mnt/Conv-TasNet/Conv_TasNet_Pytorch/data/audio_scp_8k/test/clean/tt_mix.scp', help='Path to mix scp file.')
    parser.add_argument(
        '-txt_scp_spk1', type=str, default='/mnt/Conv-TasNet/Conv_TasNet_TSE/text/test/test_spk1.csv', help='Path to text scp file.')
    parser.add_argument(
        '-txt_scp_spk2', type=str, default='/mnt/Conv-TasNet/Conv_TasNet_TSE/text/test/test_spk2.csv', help='Path to text scp file.')
    parser.add_argument(
        '-yaml', type=str, default='/mnt/Conv-TasNet/Conv_TasNet_TSE/config/train_mean_film_param.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='/mnt/Conv-TasNet/Conv_TasNet_TSE/Conv-TasNet-mean_film_1/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./val_result', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    print("gpuid:",gpuid)
    separation=Separation(args.mix_scp,args.txt_scp_spk1, args.txt_scp_spk2, args.yaml, args.model, gpuid)
    separation.inference(args.save_path)


if __name__ == "__main__":
    main()