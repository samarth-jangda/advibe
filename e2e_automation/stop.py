# The following code is used to delete the node in the
# e2e server once the model and data part is build

import argparse

parser=argparse.ArgumentParser(description="",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--authorization-config", type=str, default="", required=True,
                    help="")
