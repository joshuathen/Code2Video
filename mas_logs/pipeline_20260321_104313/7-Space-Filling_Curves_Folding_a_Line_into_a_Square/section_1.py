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

class Section1Scene(TeachingScene):
    def get_hilbert_curve(self, order, size):
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

        n = 2**order
        pts = []
        for d in range(n*n):
            x, y = d2xy(n, d)
            # Map coordinates to be centered around zero with the given total size
            px = (x / (n - 1) - 0.5) * size if n > 1 else 0
            py = (y / (n - 1) - 0.5) * size if n > 1 else 0
            pts.append(np.array([px, py, 0]))
        return pts

    def construct(self):
        # Initial Setup
        title_text = "The Dimensionality Paradox"
        lecture_lines_text = [
            "Can a one-dimensional line cover a two-dimensional square?",
            "Intuitively, a line is thin and lacks area.",
            "However, a dense map can fill every point.",
            "Imagine a snail leaving a trail across the floor.",
            "Can its path touch every single point in space?"
        ]
        self.setup_layout(title_text, lecture_lines_text)

        # Helper for color highlighting
        def highlight_lecture(idx):
            animations = []
            for i, line in enumerate(self.lecture):
                if i == idx:
                    animations.append(line.animate.set_color(YELLOW))
                else:
                    animations.append(line.animate.set_color(WHITE))
            self.play(*animations, run_time=0.5)

        # === Animation for Lecture Line 1 ===
        highlight_lecture(0)
        line_1d = Line(LEFT, RIGHT, color="#FFFFFF")
        # Fix for Issue 33: Adjusted scale factor to 1.5
        self.place_in_area(line_1d, "A2", "A5", scale_factor=1.5)
        
        square_2d = Square(side_length=3.0, color="#FFFF00")
        self.place_in_area(square_2d, "C2", "F5")
        
        self.play(Create(line_1d))
        self.play(Create(square_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        highlight_lecture(1)
        thickness_label = Text("Thickness: 0", font_size=24, color="#FFFFFF")
        # Fix for Issue 31: Using area placement for multi-word label
        self.place_in_area(thickness_label, 'B2', 'B5', scale_factor=0.8)
        
        # Simulate zoom into the line to emphasize zero thickness
        self.play(
            line_1d.animate.scale(2.5), 
            Write(thickness_label)
        )
        self.wait(1)
        self.play(
            line_1d.animate.scale(1/2.5), 
            FadeOut(thickness_label)
        )

        # === Animation for Lecture Line 3 ===
        highlight_lecture(2)
        # Begin winding path: Hilbert Order 1 (Basic U-shape)
        pts_l1 = self.get_hilbert_curve(1, 2.5)
        # Shift points to the center of the pre-placed square
        pts_l1 = [p + square_2d.get_center() for p in pts_l1]
        path_l1 = VMobject(color="#FFFFFF").set_points_as_corners(pts_l1)
        
        self.play(Create(path_l1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        highlight_lecture(3)
        # Snail leaving a thick trail: Hilbert Order 3
        pts_l3 = self.get_hilbert_curve(3, 2.5)
        pts_l3 = [p + square_2d.get_center() for p in pts_l3]
        trail = VMobject(color="#00FF00").set_points_as_corners(pts_l3)
        trail.set_stroke(width=10)
        
        # 'Quantum Snail' dot
        snail = Dot(color="#00FF00").scale(1.5)
        snail.move_to(pts_l3[0])
        
        self.play(FadeOut(path_l1), FadeIn(snail))
        self.play(Create(trail), snail.animate.move_to(pts_l3[-1]), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        highlight_lecture(4)
        # Final Question Mark Overlay
        question_mark = Text("?", font_size=144, color="#FFFFFF")
        # Fix for Issue 32: Adjusted area to avoid obstructing curve and adjusted scale
        self.place_in_area(question_mark, 'D3', 'E4', scale_factor=0.9)
        self.play(Write(question_mark))
        self.wait(2)
