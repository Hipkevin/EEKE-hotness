

def training_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--cv", type=float, default=0.2,
                        help="Cross validation rate.")
    parser.add_argument("--weight_decay", type=float, default=2e-4,
                        help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    parser.add_argument("--word_emb_dim", type=int, default=50,
                       help='Dimension of word embedding.')
    parser.add_argument("--window_size", type=int, default=3,
                       help='Time series prediction observation window size.')

    parser.add_argument("--att_head_num", type=int, default=5,
                        help='Heads number of Multi-head attention mechanism')

    parser.add_argument("--model", type=str, default="GCN",
                        help="Model Type.")
