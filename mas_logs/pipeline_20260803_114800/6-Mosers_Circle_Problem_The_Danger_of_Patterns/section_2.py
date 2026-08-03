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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        lecture_lines = [
            "With 1 point, we have 1 region.",
            "With 2 points, we get 2 regions.",
            "With 3 points, the count reaches 4.",
            "For 4 and 5 points, it's 8 and 16.",
            "The pattern seems to double every time!"
        ]
        self.setup_layout("The Seductive Pattern", lecture_lines)

        # Initially set opacity low for lecture progression
        for line in self.lecture:
            line.set_opacity(0.3)

        # Assets
        points_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/points.svg"
        
        # Create Table Elements
        # === Animation for Lecture Line 1 ===
        header_n = MathTex("n", color=WHITE)
        header_r = MathTex("R", color=WHITE)
        self.place_at_grid(header_n, "A2", scale_factor=0.8)
        self.place_at_grid(header_r, "A4", scale_factor=0.8)
        
        # Separator lines
        line_h = Line(self.grid["A1"], self.grid["A6"], color=WHITE).shift(DOWN * 0.4)
        line_v = Line(self.grid["A3"], self.grid["F3"], color=WHITE).shift(LEFT * 0.5)

        # Rows data
        n_vals = [MathTex(str(i), color=WHITE) for i in range(1, 6)]
        r_vals = [MathTex(str(2**(i-1)), color=WHITE) for i in range(1, 6)]
        
        # Create groups of icons for the 'R' column to represent the values
        # This addresses Issue 23
        icon_groups = []
        for val in [1, 2, 4, 8, 16]:
            # Scale icons based on quantity to fit in cell
            icon_scale = 0.15 if val <= 4 else 0.1
            icons = VGroup(*[SVGMobject(points_asset_path).set_height(icon_scale).set_color(WHITE) for _ in range(val)])
            if val > 1:
                icons.arrange_in_grid(rows=None, cols=4 if val > 4 else 2, buff=0.05)
            icon_groups.append(icons)

        # Positioning table values
        grid_rows = ["B", "C", "D", "E", "F"]
        for i, (n_tex, r_tex, icons) in enumerate(zip(n_vals, r_vals, icon_groups)):
            self.place_at_grid(n_tex, f"{grid_rows[i]}2", scale_factor=0.8)
            self.place_at_grid(r_tex, f"{grid_rows[i]}4", scale_factor=0.8)
            self.place_at_grid(icons, f"{grid_rows[i]}5", scale_factor=1.0)

        # Formula positioning (Issue 27)
        formula = MathTex(r"R = 2^{n-1}", color=GREEN)
        self.place_in_area(formula, "C5", "E6", scale_factor=1.0)

        # Animation Sequence
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1.0))
        self.play(Create(header_n), Create(header_r), Create(line_h), Create(line_v))
        self.play(Write(n_vals[0]), Write(r_vals[0]), FadeIn(icon_groups[0]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_opacity(1.0))
        self.play(Write(n_vals[1]), Write(r_vals[1]), FadeIn(icon_groups[1]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_opacity(1.0))
        self.play(Write(n_vals[2]), Write(r_vals[2]), FadeIn(icon_groups[2]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_opacity(1.0))
        self.play(
            Write(n_vals[3]), Write(r_vals[3]), FadeIn(icon_groups[3]),
            Write(n_vals[4]), Write(r_vals[4]), FadeIn(icon_groups[4])
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_opacity(1.0).set_color(GREEN))
        self.play(Write(formula))
        self.play(Indicate(formula))
        self.wait(2)
