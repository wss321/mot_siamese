from eval.eval_mot import eval, main

if __name__ == '__main__':
    eval(r'E:\PyProjects\datasets/MOT16',
         r"./output/", show=True, save_video=True, budget=32, iou_th=0.2, area_rate_th=2)

    seqs_str = '''
                    MOT16-02
                    MOT16-05
                    MOT16-09
                    MOT16-11
                    MOT16-13
                    '''
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root='./data/MOT16/train',
         seqs=seqs,
         exp_name='mot16_val',
         show_image=False)
