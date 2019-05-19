import argparse
import easydict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sampler_batch_size', default=256,
                        help='Batch size to use in the box sampler', type=int)
    parser.add_argument('-num_pos', default=0, help='Number of positive examples', type=int)
    parser.add_argument('-sampler_high_thresh', default=0.75,
                        help='Boxes IoU greater than this are considered positives', type=float)
    parser.add_argument('-sampler_low_thresh', default=0.4,
                        help='Boxes with IoU less than this are considered negatives', type=float)
    parser.add_argument('-train_remove_outbounds_boxes', default=1,
                        help='Whether to ignore out-of-bounds boxes for sampling at training time', type=int)

    # Model
    parser.add_argument('-std', default=0.01, help='std for init', type=int)
    parser.add_argument('-init_weights', default=1,
                        help='Whether to initialize weight for rpn and two final fc', type=int)

    # Loss function
    parser.add_argument('-mid_box_reg_weight', default=0.1, help='Weight for box regression in the RPN', type=float)
    parser.add_argument('-mid_objectness_weight', default=0.5, help='Weight for box classification in the RPN',
                        type=float)
    parser.add_argument('-end_box_reg_weight', default=0.1, help='Weight for box regression in the recognition network',
                        type=float)
    parser.add_argument('-end_objectness_weight', default=0.5,
                        help='Weight for box classification in the recognition network', type=float)
    parser.add_argument('-weight_decay', default=1e-5, help='L2 weight decay penalty strength', type=float)
    parser.add_argument('-box_reg_decay', default=5e-5,
                        help='Strength of pull that boxes experience towards their anchor', type=float)

    # Data input settings
    parser.add_argument('-image_size', default=1720, help='which fold to use', type=int)
    parser.add_argument('-dtp_train', default=1, help='Whether or not to use DTP in train', type=int)

    # Optimization
    parser.add_argument('-learning_rate', default=2e-4, help='learning rate to use', type=float)
    parser.add_argument('-reduce_lr_every', default=12000, help='reduce learning rate every x iterations', type=int)
    parser.add_argument('-beta1', default=0.9, help='beta1 for adam', type=float)
    parser.add_argument('-beta2', default=0.999, help='beta2 for adam', type=float)
    parser.add_argument('-epsilon', default=1e-8, help='epsilon for smoothing', type=float)
    parser.add_argument('-max_iters', default=10000, help='Number of iterations to run; -1 to run forever', type=int)
    parser.add_argument('-pretrained',  action='store_true', help='Load model from a checkpoint instead of random initialization.')
    parser.add_argument('-model_path', help='path to the pretrained model')
    # Model checkpointing
    parser.add_argument('-eval_every', default=200, help='How often to test on validation set', type=int)

    # Test-time model options (for evaluation)
    parser.add_argument('-test_rpn_nms_thresh', default=0.4,
                        help='Test-time NMS threshold to use in the RPN', type=float)
    parser.add_argument('-max_proposals', default=-1,
                        help='Number of region proposal to use at test-time', type=int)
    parser.add_argument('-score_nms_overlap', default=0.5,
                        help='NMS overlap using box scores in postprocessing', type=float)
    parser.add_argument('-score_threshold', default=0.65,
                        help='score threshold using box scores in postprocessing', type=float)
    parser.add_argument('-dtp_test', default=1, help='Whether or not to use DTP in test', type=int)
    parser.add_argument('-test_batch_size', default=128, help='Whether or not to use DTP', type=int)

    # Visualization
    parser.add_argument('-print_every', default=10, help='How often to print the latest images training loss.',
                        type=int)
    parser.add_argument('-out_path', default='out', help='output dir for intermediate results')
    # Misc
    parser.add_argument('-save_id', default='', help='an id identifying this run/job')
    parser.add_argument('-quiet', default=0, help='run in quiet mode, no prints', type=int)
    parser.add_argument('-verbose', default=0, help='print info in localization layer in test', type=int)
    parser.add_argument('-gpu', default=1, help='use gpu or not.', type=int)
    parser.add_argument('-clip_final_boxes', default=1, help='Whether to clip final boxes to image boundar', type=int)

    args = parser.parse_args()

    return easydict.EasyDict(vars(args))
