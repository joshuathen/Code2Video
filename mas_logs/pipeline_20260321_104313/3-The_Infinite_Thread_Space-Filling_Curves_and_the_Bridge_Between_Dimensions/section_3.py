from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def d2xy(self, n, d):
        def rot(n, x, y, rx, ry):
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                return y, x
            return x, y

        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    def get_hilbert_curve(self, order, color, size=4.0):
        n = 2**order
        points = []
        for d in range(n * n):
            ix, iy = self.d2xy(n, d)
            # Scale coordinates to fit size and center them in the local frame
            if n > 1:
                # Map [0, n-1] to [-0.5, 0.5] then multiply by size
                x = (ix / (n - 1) - 0.5) * size
                y = (iy / (n - 1) - 0.5) * size
            else:
                x, y = 0, 0
            points.append(np.array([x, y, 0]))
        
        curve = VMobject()
        curve.set_points_as_corners(points)
        curve.set_color(color)
        curve.set_stroke(width=3)
        return curve

    def construct(self):
        # Initial lecture setup
        lecture_lines = [
            "The Hilbert curve begins with a simple four-point path.",
            "At step two, the path replicates and rotates.",
            "Each iteration replaces segments with smaller sub-curves.",
            "The path remains a single, non-self-intersecting continuous line.",
            "Watch the square become increasingly crowded with segments."
        ]
        self.setup_layout("Constructing the Hilbert Curve", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a 1st-order Hilbert curve in #FFFFFF connecting four points in a 2x2 grid.
        # Line 1 text is already White by default.
        curve_obj = self.get_hilbert_curve(1, WHITE)
        self.place_in_area(curve_obj, "A1", "F6", scale_factor=1.0)
        self.play(Create(curve_obj), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the curve into its 2nd-order form by replicating and rotating the base shape in #00FF00.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        curve2 = self.get_hilbert_curve(2, "#00FF00")
        self.place_in_area(curve2, "A1", "F6", scale_factor=1.0)
        self.play(Transform(curve_obj, curve2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the transition to the 3rd-order curve, replacing each segment with a sub-curve in #ADFF2F.
        self.play(self.lecture[2].animate.set_color("#ADFF2F"))
        curve3 = self.get_hilbert_curve(3, "#ADFF2F")
        self.place_in_area(curve3, "A1", "F6", scale_factor=1.0)
        self.play(Transform(curve_obj, curve3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A white glow #FFFFFF traces the entire path to show it is a single continuous line.
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        # We trace a white line over the existing curve_obj (which now looks like curve3)
        glow_path = curve_obj.copy().set_color(WHITE).set_stroke(width=8, opacity=0.8)
        self.play(ShowPassingFlash(glow_path, run_time=3, time_width=0.4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Rapidly display the 4th and 5th iterations as the path fills the square.
        self.play(self.lecture[4].animate.set_color("#ADFF2F"))
        
        # Iteration 4
        curve4 = self.get_hilbert_curve(4, "#ADFF2F")
        self.place_in_area(curve4, "A1", "F6", scale_factor=1.0)
        self.play(Transform(curve_obj, curve4), run_time=1.5)
        
        # Iteration 5
        curve5 = self.get_hilbert_curve(5, "#ADFF2F")
        self.place_in_area(curve5, "A1", "F6", scale_factor=1.0)
        self.play(Transform(curve_obj, curve5), run_time=1.5)
        
        self.wait(3)
