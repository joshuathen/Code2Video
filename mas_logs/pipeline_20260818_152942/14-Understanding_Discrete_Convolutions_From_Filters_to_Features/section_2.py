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
        lecture_lines = [
            "Convolution blends two functions together.",
            "We flip, slide, multiply, and sum.",
            "The kernel slides over the input.",
            "Each overlap creates one output value.",
            "This forms a new feature map."
        ]
        self.setup_layout("Defining Discrete Convolution", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # 1. Define formula (f * g)[n] = Σ f[k]g[n-k]
        formula = MathTex(r"(f * g)[n] = \sum_{k} f[k] \cdot g[n-k]", color="#00FFFF")
        # Included asset placeholder using SVGMobject for SVG files
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(formula, "A1", "B6", scale_factor=0.6)
        self.play(Write(formula))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 2. Show function f and kernel g as arrays.
        f_arr = Matrix([[1, 2, 3, 4, 5]])
        g_arr = Matrix([[0, 1, 0]])
        f_label = Text("f", font_size=20).next_to(f_arr, UP)
        g_label = Text("g", font_size=20).next_to(g_arr, UP)
        content = VGroup(f_label, f_arr, g_label, g_arr).arrange(DOWN)
        self.place_in_area(content, "C1", "D6", scale_factor=0.8)
        self.play(Create(f_arr), Write(f_label), Create(g_arr), Write(g_label))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 3. Flip kernel g horizontally, show in #FFFF00.
        g_flipped = Matrix([[0, 1, 0]])
        g_flipped.set_color("#FFFF00")
        self.play(Transform(g_arr, g_flipped))
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 4. Slide kernel g over f, calculate sum-product.
        brace = Brace(f_arr, DOWN)
        text_slide = Text("Slide", font_size=18).next_to(brace, DOWN)
        self.play(Create(brace), Write(text_slide))
        self.play(g_arr.animate.shift(RIGHT * 0.5), run_time=1)
        self.lecture[3].set_color("#7FFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 5. Display result array in #7FFF00.
        res_arr = Matrix([[0, 1, 2, 1, 0]])
        res_arr.set_color("#7FFF00")
        res_label = Text("Result", font_size=20).next_to(res_arr, UP)
        res_group = VGroup(res_label, res_arr)
        # Included asset placeholder
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(res_group, "E1", "F6", scale_factor=0.7)
        self.play(Write(res_label), Create(res_arr))
        self.lecture[4].set_color("#7FFF00")
        self.wait(2)
