from eval.eval_mot import eval, main
import os

if __name__ == '__main__':
    seqs = os.listdir(r'E:\PyProjects\datasets/MOT17/train')
    print(seqs)
    eval(r'E:\PyProjects\datasets/MOT17',
         r"./output/", show=True, save_video=True, budget=16, iou_th=0.0, area_rate_th=2)
    main(data_root=r'E:\PyProjects\datasets/MOT17/train',
         seqs=seqs,
         exp_name='mot17_val',
         show_image=False)
