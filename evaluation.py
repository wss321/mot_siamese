from eval.eval_mot import tracking, eval
import argparse

parser = argparse.ArgumentParser(description='Evaluating')
parser.add_argument('--data_dir', default='../datasets/MOT16/train', type=str, help='dataset path')
parser.add_argument('--output_dir', default='./output/', type=str, help='./test_data')
parser.add_argument('--det_file', default='det.txt', type=str, help='detection file')
parser.add_argument('--e_type', default='train', type=str, help='type of running')
parser.add_argument('--model_path', default=r"E:\PyProjects\MOT\tracker\siamese\net_last.pth", type=str,
                    help='siamese model path')
parser.add_argument('--show', action='store_true', help='show tracking result')
parser.add_argument('--show_kalman', action='store_true', help='show kalman predict result')
parser.add_argument('--save_video', action='store_true', help='save video')
parser.add_argument('--crowed', action='store_true', help='whether dataset is crowed or not')
parser.add_argument('--score_th', default=0.3, type=float, help='min bbox score threshold')
parser.add_argument('--num_classes', default=751, type=int, help='min bbox score threshold')
parser.add_argument('--age', default=30, type=int, help='max time since last update of a track')
parser.add_argument('--sim_th', default=0.9, type=float, help='similarity threshold')
parser.add_argument('--budget', default=32, type=int, help='max number of image patches to compute similarity')
args = parser.parse_args()

if __name__ == '__main__':
    tracking(args)
    eval(args)
    # python evaluation.py --show --show_kalman --data_dir ../datasets/MOT16/train --score_th 0.3 --age 40 --sim_th 0.9 --budget 16
    # python evaluation.py --show --show_kalman --data_dir ../datasets/MOT17/train --score_th 0.3 --age 40 --sim_th 0.9 --budget 16
    # python evaluation.py --show --show_kalman --data_dir ../datasets/Crowd_PETS09/S2/L1 --model_path ./tracker/siamese/net_last.pth--score_th 0.4 --age 40 --sim_th 0.9 --budget 16
    # python evaluation.py --show --show_kalman --data_dir ../datasets/Crowd_PETS09/S2/L2 --score_th 0.0 --age 40 --sim_th 0.9 --budget 16
    # python evaluation.py --show --show_kalman --data_dir ../datasets/Crowd_PETS09/S2/L3 --score_th 0.0 --age 40 --sim_th 0.9 --budget 16
