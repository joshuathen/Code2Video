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
        title = "1D Graphical Walkthrough"
        lines = [
            "Let’s visualize two pulses colliding and passing through.",
            "As they start overlapping, the output value begins growing.",
            "The peak occurs when the signals achieve maximum overlap.",
            "The output fades as the signals move apart.",
            "The result represents the mathematical area of their overlap."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_X = "#67ABFF"  # Light Blue
        COLOR_H = "#FFE066"  # Light Yellow
        COLOR_Y = "#FF6767"  # Light Red
        COLOR_AXIS = WHITE

        # Origins for graphs
        # Top origin for x[k] and h[n-k]
        top_origin = (self.grid["B3"] + self.grid["B4"]) / 2
        # Bottom origin for y[n]
        bottom_origin = (self.grid["E3"] + self.grid["E4"]) / 2

        # Parameters
        W = 1.0  # Pulse Width
        H = 0.8  # Pulse Height
        
        # Axes
        top_axis = Line(self.grid["B1"], self.grid["B6"], color=COLOR_AXIS)
        bottom_axis = Line(self.grid["E1"], self.grid["E6"], color=COLOR_AXIS)
        
        # Labels - Fix based on Issues 26, 27, 28
        x_label = Text("x[k]", font_size=20, color=COLOR_X)
        self.place_at_grid(x_label, "A3") # Issue 26: A2 -> A3
        
        h_label = Text("h[n-k]", font_size=20, color=COLOR_H)
        self.place_at_grid(h_label, "A6") # Issue 28: A5 -> A6
        
        y_label = Text("y[n]", font_size=20, color=COLOR_Y)
        self.place_at_grid(y_label, "D3") # Issue 27: D2 -> D3

        # x[k] static pulse [0, W]
        x_rect = Rectangle(width=W, height=H, fill_opacity=0.4, color=COLOR_X, stroke_width=2)
        x_rect.move_to(top_origin + RIGHT * (W / 2) + UP * (H / 2))

        # Value tracker for shift n
        n_tracker = ValueTracker(-1.2)

        # h[n-k] moving pulse [n-W, n]
        # h[k] is a rect from 0 to W.
        # h[-k] is a rect from -W to 0.
        # h[n-k] is shifted by n, so it's a rect from n-W to n.
        h_rect = Rectangle(width=W, height=H, fill_opacity=0.4, color=COLOR_H, stroke_width=2)
        h_rect.add_updater(lambda m: m.move_to(top_origin + RIGHT * (n_tracker.get_value() - W / 2) + UP * (H / 2)))

        # Resulting output path y[n]
        # Convolution of two rects of width W is a triangle peaking at n=W with height W.
        # Function: y(n) = n for 0<=n<=W, y(n) = 2W-n for W<=n<=2W.
        output_path = VMobject(color=COLOR_Y, stroke_width=4)
        
        def update_output_path(m):
            n = n_tracker.get_value()
            pts = [bottom_origin + LEFT * 1.2, bottom_origin]
            if n > 0:
                if n <= W:
                    # Rising slope
                    pts.append(bottom_origin + RIGHT * n + UP * n)
                elif n <= 2 * W:
                    # Peak and falling slope
                    pts.append(bottom_origin + RIGHT * W + UP * W)
                    pts.append(bottom_origin + RIGHT * n + UP * (2 * W - n))
                else:
                    # Past the interaction
                    pts.append(bottom_origin + RIGHT * W + UP * W)
                    pts.append(bottom_origin + RIGHT * (2 * W))
                    pts.append(bottom_origin + RIGHT * n)
            m.set_points_as_corners(pts)

        output_path.add_updater(update_output_path)

        # === Animation for Lecture Line 1 ===
        # Let’s visualize two pulses colliding and passing through.
        self.lecture[0].set_color(COLOR_X)
        self.add(top_axis, bottom_axis, x_label, h_label, y_label, x_rect, h_rect, output_path)
        self.play(n_tracker.animate.set_value(0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # As they start overlapping, the output value begins growing.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_Y)
        self.play(n_tracker.animate.set_value(0.5 * W), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The peak occurs when the signals achieve maximum overlap.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_Y)
        self.play(n_tracker.animate.set_value(W), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The output fades as the signals move apart.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_Y)
        self.play(n_tracker.animate.set_value(2.2 * W), run_time=4)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The result represents the mathematical area of their overlap.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_Y)
        # Highlight the triangle
        self.play(output_path.animate.set_stroke(width=6), run_time=1)
        self.play(output_path.animate.set_stroke(width=4), run_time=1)
        self.wait(2)
