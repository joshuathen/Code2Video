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
        # 1. Setup layout with script content
        lecture_lines = [
            "For odd functions, all cosine terms vanish to zero.",
            "We are left with a pure sine series representation.",
            "Integration only requires half the period, then double it."
        ]
        self.setup_layout("Deriving the Pure Sine Series", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in Red to match vanishing terms
        self.play(self.lecture[0].animate.set_color("#FF0000"))

        # Odd Square Wave (Visual representative of an odd function)
        # Starting at C3 as per VideoCritic fix (Issue 37)
        wave = VMobject(color=BLUE)
        wave_points = [
            [-1.0, 0.4, 0], [-0.5, 0.4, 0], [-0.5, -0.4, 0], [0, -0.4, 0], 
            [0, 0.4, 0], [0.5, 0.4, 0], [0.5, -0.4, 0], [1.0, -0.4, 0]
        ]
        wave.set_points_as_corners(wave_points)
        self.place_at_grid(wave, "C3", scale_factor=0.8)
        
        # Cosine Filter Box (#FFFFFF) at B4-D6 as per VideoCritic fix (Issue 37)
        filter_box = Rectangle(width=2.5, height=2.2, color=WHITE)
        filter_label = Text("Cosine Filter", font_size=24, color=WHITE)
        filter_group = VGroup(filter_box, filter_label).arrange(UP, buff=0.1)
        self.place_in_area(filter_group, "B4", "D6")

        self.play(Create(filter_group), Create(wave))
        self.wait(0.5)

        # Wave enters the box to be filtered
        box_center = self.grid["C5"] # Center of B4-D6 area
        self.play(wave.animate.move_to(box_center).set_opacity(0.4), run_time=1.5)

        # Cosine terms (an) ejected from the filter (#FF0000)
        an_symbols = VGroup(*[MathTex("a_n", color="#FF0000") for _ in range(3)])
        for sym in an_symbols:
            sym.move_to(box_center)
        
        # Paths for ejection - exiting the filter box bounds upwards and rightwards
        ejection_targets = [
            box_center + UP*1.8,
            box_center + UP*1.8 + RIGHT*1.0,
            box_center + RIGHT*1.8
        ]
        
        ejection_anims = []
        for i in range(3):
            ejection_anims.append(
                Succession(
                    FadeIn(an_symbols[i]),
                    an_symbols[i].animate.move_to(ejection_targets[i]).set_opacity(0),
                    run_time=1.0
                )
            )
        
        self.play(AnimationGroup(*ejection_anims, lag_ratio=0.4))

        # an = 0 text appears in bold (#FF0000) at E6 as per VideoCritic fix (Issue 38)
        an_zero = MathTex(r"a_n = 0", color="#FF0000", font_size=48)
        self.place_at_grid(an_zero, "E6")
        self.play(Write(an_zero))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition color: revert Line 1, maintain Line 2 white
        self.play(
            self.lecture[0].animate.set_color(WHITE)
        )

        # Pure sine series representation at A3-A6 as per VideoCritic fix (Issue 39)
        sine_series = MathTex(
            r"f(x) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right)",
            color=WHITE, font_size=36
        )
        self.place_in_area(sine_series, "A3", "A6")
        
        self.play(FadeIn(sine_series))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3 in Yellow to match bn highlight
        self.play(
            self.lecture[2].animate.set_color("#FFFF00")
        )

        # Formula for bn with 2/L and limits highlighted (#FFFF00) at F3-F6 as per VideoCritic fix (Issue 39)
        bn_formula = MathTex(
            r"b_n = ", r"\frac{2}{L}", r" \int", r"_{0}", r"^{L}", r" f(x) \sin\left(\frac{n\pi x}{L}\right) dx",
            color=WHITE, font_size=36
        )
        # Highlight requested parts in yellow (#FFFF00)
        bn_formula[1].set_color("#FFFF00") # 2/L
        bn_formula[3].set_color("#FFFF00") # 0
        bn_formula[4].set_color("#FFFF00") # L
        
        self.place_in_area(bn_formula, "F3", "F6")
        
        self.play(FadeIn(bn_formula))
        self.wait(2)
