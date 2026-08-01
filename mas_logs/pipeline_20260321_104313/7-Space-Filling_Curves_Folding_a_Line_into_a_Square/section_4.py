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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "Mathematical Mapping: [0, 1] → [0, 1]²",
            [
                "We map intervals on a line to 2D coordinates.",
                "The function is continuous, never lifting the pen.",
                "It is surjective, hitting every point in the square."
            ]
        )

        # Hilbert Curve Logic
        def get_hilbert_points(order):
            n = 2**order
            def rot(n, x, y, rx, ry):
                if ry == 0:
                    if rx == 1:
                        x = n - 1 - x
                        y = n - 1 - y
                    return y, x
                return x, y

            def d2xy(n, d):
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

            points = []
            for d in range(n * n):
                x, y = d2xy(n, d)
                points.append(np.array([x, y, 0]))
            return points, n

        # Parameters
        order = 4
        h_points, n_side = get_hilbert_points(order)
        
        # Grid positions for mobjects
        # Square Area: B3 to D5 (providing clearance for labels)
        # Line Area: E3 to E5 (providing clearance for labels)
        sq_center = (self.grid["B3"] + self.grid["D5"]) / 2
        sq_side = self.grid["D5"][0] - self.grid["B3"][0] # 2.0
        line_center = (self.grid["E3"] + self.grid["E5"]) / 2
        line_width = self.grid["E5"][0] - self.grid["E3"][0] # 2.0

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        # Create elements with matching color (Yellow)
        square = Square(side_length=sq_side, color="#FFFF00")
        self.place_in_area(square, "B3", "D5")
        
        line_base = Line(start=LEFT * (line_width / 2), end=RIGHT * (line_width / 2), color="#FFFF00")
        self.place_in_area(line_base, "E3", "E5")

        # Normalize Hilbert points to fit the new square position/scale
        norm_points = []
        for p in h_points:
            # Shift 0..n-1 to -0.5..0.5, scale by sq_side, then shift to sq_center
            nx = (p[0] / (n_side - 1) - 0.5) * sq_side
            ny = (p[1] / (n_side - 1) - 0.5) * sq_side
            norm_points.append(np.array([nx, ny, 0]) + sq_center)

        hilbert_path_ref = VMobject()
        hilbert_path_ref.set_points_as_corners(norm_points)

        # Red Dots per instruction
        dot_line = Dot(color="#FF0000", radius=0.08).move_to(line_base.get_start())
        dot_sq = Dot(color="#FF0000", radius=0.08).move_to(norm_points[0])

        # Labels
        label_t = Text("t", font_size=20, color="#FFFFFF")
        label_ft = Text("f(t)", font_size=20, color="#FFFFFF")
        
        # Initial label placement
        label_t.next_to(dot_line, DOWN, buff=0.1)
        label_ft.next_to(dot_sq, UP, buff=0.1)
        
        # Updaters for labels to follow dots
        label_t.add_updater(lambda m: m.next_to(dot_line, DOWN, buff=0.1))
        label_ft.add_updater(lambda m: m.next_to(dot_sq, UP, buff=0.1))

        self.play(Create(square), Create(line_base))
        self.play(FadeIn(dot_line), FadeIn(dot_sq), FadeIn(label_t), FadeIn(label_ft))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#5555FF")
        
        # Track the progress (0 to 1) for the mapping
        progress = ValueTracker(0)
        
        # Set up dot updaters for smooth movement along the 1D and 2D paths
        dot_line.add_updater(lambda d: d.move_to(line_base.point_from_proportion(progress.get_value())))
        dot_sq.add_updater(lambda d: d.move_to(hilbert_path_ref.point_from_proportion(progress.get_value())))
        
        # Trace path (Blue to match Lecture Line 2)
        trace = TracedPath(dot_sq.get_center, stroke_color="#5555FF", stroke_width=2.5)
        self.add(trace)

        self.play(progress.animate.set_value(1), run_time=6, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Surjective fill (Green to match Lecture Line 3)
        surj_fill = Square(side_length=sq_side, color="#00FF00", fill_opacity=0.2, stroke_width=0)
        self.place_in_area(surj_fill, "B3", "D5")
        
        self.play(FadeIn(surj_fill))
        self.play(Indicate(square, color="#00FF00"))
        self.wait(2)

        # Cleanup updaters
        label_t.clear_updaters()
        label_ft.clear_updaters()
        dot_line.clear_updaters()
        dot_sq.clear_updaters()
