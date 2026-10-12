"""A window showing what the board camera sees.

    python -m sim.camera_view                      # one frame of the opening position
    python -m sim.camera_view --fen "..." --out shot.png
    python -m sim.camera_view --elevation 90       # straight down, for comparison
    python -m sim.camera_view --azimuth 0          # from the h-file side
    python main.py --camera-view               # live, while the arm plays

While the arm moves the raw frame is shown, robot and all.  Once a move
finishes and the board is static, the frame is redrawn with the square grid and
the detector's reading on top.
"""

import argparse
import pathlib

import cv2
import numpy as np

from . import board_layout as L
from .perception import (BoardCamera, MarkerPerception, annotate,
                         grid_homography)

FALLBACK_DIR = pathlib.Path("camera_view")


class CameraView:
    """Shows and/or saves camera frames.

    Falls back to writing PNGs when OpenCV has no GUI support, as with
    opencv-python-headless.
    """

    def __init__(self, sim, window="Kasparobot board camera", scale=1.0,
                 overlay=True, save_dir=None, resolution=None, reader=None,
                 homography=False):
        self.sim = sim
        self.window = window
        self.scale = scale
        self.overlay = overlay
        self.homography = homography
        self.resolution = resolution or L.CAM_RES
        self.save_dir = pathlib.Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.camera = BoardCamera.from_sim(sim, resolution=self.resolution)
        self.reader = reader or MarkerPerception(self.camera)
        # Only the marker detector needs the crown discs.  Without a usable
        # detector, draw the grid but do not pretend to read the board.
        self.detect = (not getattr(self.reader, "requires_markers", True)
                       or bool(getattr(sim, "markers", True)))
        self._can_show = True
        self._warned = False
        self._n = 0

    def _camera_for(self, rgb):
        cam = self.camera
        return cam if rgb.shape[0] == cam.res else cam.resized(rgb.shape[0])

    def frame(self, overlay=None):
        """Render one frame, optionally with the detector's reading drawn on."""
        rgb = self.sim.capture(self.resolution, self.resolution, park=False)
        use_overlay = self.overlay if overlay is None else overlay
        found = []
        if use_overlay:
            found = self.reader.detect(rgb) if self.detect else []
            img = annotate(rgb, found, self.camera)
        else:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.homography:
            img = np.hstack([img, self.rectified(rgb, found, img.shape[0])])
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        return img

    def rectified(self, rgb, detections=(), size=None):
        """The camera image warped flat through the board homography.

        The space the square mapping works in.  Pieces smear outwards because
        solid objects are being flattened onto the board plane, which is why
        the mapping uses the foot of each box rather than its centre.
        """
        size = size or self.resolution
        cam = self._camera_for(rgb)
        margin = 0.75                     # room for the edge labels
        span = 8 + 2 * margin
        step = size / span
        scale = np.array([[step, 0.0, margin * step],
                          [0.0, step, margin * step],
                          [0.0, 0.0, 1.0]])
        warp = scale @ grid_homography(cam)
        panel = cv2.warpPerspective(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    warp, (size, size))

        to_px = lambda j, i: (int(round((j + margin) * step)),
                              int(round((i + margin) * step)))
        for k in range(9):
            cv2.line(panel, to_px(k, 0), to_px(k, 8), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.line(panel, to_px(0, k), to_px(8, k), (0, 0, 0), 1, cv2.LINE_AA)
        for k, file in enumerate(L.FILES):
            cv2.putText(panel, file, to_px(k + 0.38, 8.42), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * step / 70, (235, 235, 235), 1, cv2.LINE_AA)
        for k, rank in enumerate(reversed(L.RANKS)):
            cv2.putText(panel, rank, to_px(-0.42, k + 0.62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * step / 70, (235, 235, 235), 1, cv2.LINE_AA)

        for det in detections:
            j = det.x / L.SQUARE + 4.0
            i = 4.0 - det.y / L.SQUARE
            centre = to_px(j, i)
            cv2.circle(panel, centre, max(3, int(step * 0.09)), (0, 0, 255), -1)
            label = f"{det.symbol}{det.square}"
            origin = (centre[0] + int(step * 0.14), centre[1] - int(step * 0.10))
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.42 * step / 70, colour, thick, cv2.LINE_AA)
        title = (int(step * 0.12), size - int(step * 0.18))
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(panel, "homography", title, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * step / 70, colour, thick, cv2.LINE_AA)
        return panel

    def update(self, tag=None, overlay=None, save=True):
        """Render a frame, show it, and save it if it is worth keeping.

        `save=False` is for live frames during a move: worth watching, not
        worth writing forty of.  With no window and nothing to save, the render
        is skipped.
        """
        keep = save and (self.save_dir is not None or not self._can_show)
        if not self._can_show and not keep:
            return None
        img = self.frame(overlay=overlay)
        if keep:
            target = self.save_dir or FALLBACK_DIR
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{tag or f'{self._n:04d}'}.png"), img)
            self._n += 1
        if self._can_show:
            try:
                cv2.imshow(self.window, img)
                cv2.waitKey(1)
            except cv2.error:
                self._can_show = False
                if not self._warned:
                    self._warned = True
                    print("  (this OpenCV build has no GUI window - install "
                          "opencv-python in place of opencv-python-headless "
                          f"for a live view; saving frames to "
                          f"{self.save_dir or FALLBACK_DIR}/ instead)")
                if save and not keep:      # `keep` already wrote this frame
                    target = self.save_dir or FALLBACK_DIR
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target / f"{tag or f'{self._n:04d}'}.png"), img)
                    self._n += 1
        return img

    def close(self):
        if self._can_show:
            try:
                cv2.destroyWindow(self.window)
            except cv2.error:
                pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fen", default=None, help="position to show")
    p.add_argument("--out", default="board_view.png",
                   help="where to write the frame")
    p.add_argument("--img-size", type=int, default=L.CAM_RES,
                   help="rendered frame size in pixels, square")
    p.add_argument("--azimuth", type=float, default=None,
                   help=f"degrees around the board from +x (default {L.CAM_AZIMUTH:g})")
    p.add_argument("--elevation", type=float, default=None,
                   help=f"degrees above the board plane, 90 looks straight down "
                        f"(default {L.CAM_ELEVATION:g})")
    p.add_argument("--distance", type=float, default=None,
                   help=f"metres from the board centre (default {L.CAM_DISTANCE:g})")
    p.add_argument("--plain", action="store_true",
                   help="no grid or detection overlay")
    p.add_argument("--markers", action="store_true",
                   help="put the coloured type discs back on the pieces, which "
                        "is what the marker detector reads")
    p.add_argument("--homography", action="store_true",
                   help="add the flattened board view beside the camera view")
    args = p.parse_args(argv)

    from .sim_env import ChessSim
    sim = ChessSim(fen=args.fen, verbose=False, markers=args.markers)
    if any(v is not None for v in (args.azimuth, args.elevation, args.distance)):
        sim.move_camera(args.azimuth, args.elevation, args.distance)
    view = CameraView(sim, overlay=not args.plain, resolution=args.img_size,
                      homography=args.homography)
    cv2.imwrite(args.out, view.frame())
    print(f"wrote {args.out}")
    if not args.plain and view.detect:
        print("read:", view.reader.read_fen(
            sim.capture(args.img_size, args.img_size, park=False)))
    elif not args.plain:
        print("(pieces are plain, so there is nothing for the marker detector "
              "to read - add --markers)")
    sim.close()


if __name__ == "__main__":
    main()
