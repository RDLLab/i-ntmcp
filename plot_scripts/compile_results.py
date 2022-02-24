"""Script for compiling .tsv results files into a single file """
from intmcp.run.log import compile_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str,
                        help="Dir containing the result files to compile")
    args = parser.parse_args()
    compile_results(args.result_dir)
