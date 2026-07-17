import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(config_path):
    print("=== Stage 1: train modulated INRs ===")
    from scripts.train_inr import main as t1
    t1(config_path)
    print("=== Stage 2: train mapping network g_psi ===")
    from scripts.train_mapping import main as t2
    t2(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/tiny.yaml")
