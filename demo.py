import logging
import sys

from segmentation_manager import SegmentationManager

logging.basicConfig(format="[%(module)s:%(funcName)s] %(message)s", stream=sys.stdout, level=logging.INFO)

DEMO_CT_CASE = lambda n: f"data/Case{n}_CT.nii.gz"
CONNECTIVITY = 1


def demo():
    for i in range(1, 6):
        sm = SegmentationManager(DEMO_CT_CASE(i), CONNECTIVITY)
        sm.run(post_process=True)


if __name__ == "__main__":
    demo()
