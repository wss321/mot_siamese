from eval.eval_mot import eval, main
import os

if __name__ == '__main__':
    seqs = os.listdir(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1')
    eval(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1',
         r"./output/", e_type="", show=True, save_video=True, budget=30, iou_th=0.0, area_rate_th=3, score_th=-1)
    main(data_root=r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1',
         seqs=seqs,
         exp_name='pets2009',
         show_image=False)
    # seqs = os.listdir(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2')
    # # eval(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2',
    # #      r"./output/", e_type="", show=True, save_video=True, budget=30, iou_th=0.0, area_rate_th=3, score_th=-1
    # main(data_root=r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2',
    #      seqs=seqs,
    #      exp_name='pets2009',
    #      show_image=False)
    #
    # seqs = os.listdir(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3')
    # # eval(r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3',
    # #      r"./output/", e_type="", show=True, save_video=True, budget=30, iou_th=0.0, area_rate_th=3, score_th=-1)
    # main(data_root=r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3',
    #      seqs=seqs,
    #      exp_name='pets2009',
    #      show_image=False)
