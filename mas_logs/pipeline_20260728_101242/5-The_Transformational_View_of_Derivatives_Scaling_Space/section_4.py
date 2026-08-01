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
        title = "Visualizing Magnitude and Direction"
        lines = [
            "Magnitude greater than one indicates local expansion.",
            "Magnitude less than one shows local compression.",
            "A negative derivative flips the orientation of space.",
            "Imagine an accordion stretching or squeezing.",
            "Values determine the strength and direction of transform."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_EXP = "#00FF00"
        COLOR_COMP = "#FF0000"
        COLOR_FLIP = "#BB88FF"
        COLOR_ACC = "#FFFF00"
        ASSET_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/accordion.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_EXP))
        
        inp_1 = Line(LEFT, RIGHT, color=WHITE).scale(0.5)
        out_1 = Line(LEFT, RIGHT, color=COLOR_EXP).scale(1.0)
        arr_1 = Arrow(UP*0.5, DOWN*0.5, buff=0.1, color=GRAY, stroke_width=2)
        lab_1 = MathTex("|f'(x)| > 1", color=COLOR_EXP, font_size=24)
        sub_lab_1 = Text("Expansion", font_size=16, color=COLOR_EXP).next_to(lab_1, DOWN, buff=0.05)
        lab_group_1 = VGroup(lab_1, sub_lab_1)
        acc_1 = SVGMobject(ASSET_PATH).set_color(COLOR_EXP).stretch_to_fit_width(1.5).scale(0.3)
        
        # Relative positioning before placing group
        inp_1.shift(UP*0.8)
        arr_1.next_to(inp_1, DOWN, buff=0.2)
        out_1.next_to(arr_1, DOWN, buff=0.2)
        lab_group_1.next_to(out_1, DOWN, buff=0.2)
        acc_1.next_to(out_1, UP, buff=0.1) # Place accordion on top of the output segment
        
        expansion_group = VGroup(inp_1, out_1, arr_1, lab_group_1, acc_1)
        # Issue 29 fix
        self.place_in_area(expansion_group, 'A2', 'B5', scale_factor=0.8)
        
        self.play(Create(inp_1))
        self.play(GrowArrow(arr_1))
        self.play(Create(out_1), Write(lab_group_1), FadeIn(acc_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_COMP)
        )
        
        inp_2 = Line(LEFT, RIGHT, color=WHITE).scale(0.5)
        out_2 = Line(LEFT, RIGHT, color=COLOR_COMP).scale(0.25)
        arr_2 = Arrow(UP*0.5, DOWN*0.5, buff=0.1, color=GRAY, stroke_width=2)
        lab_2 = MathTex("|f'(x)| < 1", color=COLOR_COMP, font_size=24)
        sub_lab_2 = Text("Compression", font_size=16, color=COLOR_COMP).next_to(lab_2, DOWN, buff=0.05)
        lab_group_2 = VGroup(lab_2, sub_lab_2)
        acc_2 = SVGMobject(ASSET_PATH).set_color(COLOR_COMP).stretch_to_fit_width(0.5).scale(0.3)
        
        inp_2.shift(UP*0.8)
        arr_2.next_to(inp_2, DOWN, buff=0.2)
        out_2.next_to(arr_2, DOWN, buff=0.2)
        lab_group_2.next_to(out_2, DOWN, buff=0.2)
        acc_2.next_to(out_2, UP, buff=0.1)
        
        compression_group = VGroup(inp_2, out_2, arr_2, lab_group_2, acc_2)
        # Issue 30 fix
        self.place_in_area(compression_group, 'C2', 'D5', scale_factor=0.8)
        
        self.play(Create(inp_2))
        self.play(GrowArrow(arr_2))
        self.play(Create(out_2), Write(lab_group_2), FadeIn(acc_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_FLIP)
        )
        
        inp_3 = Arrow(LEFT, RIGHT, color=WHITE, buff=0).scale(0.5)
        out_3 = Arrow(RIGHT, LEFT, color=COLOR_FLIP, buff=0).scale(0.5)
        arr_3 = Arrow(UP*0.5, DOWN*0.5, buff=0.1, color=GRAY, stroke_width=2)
        lab_3 = MathTex("f'(x) < 0", color=COLOR_FLIP, font_size=24)
        sub_lab_3 = Text("Flip", font_size=16, color=COLOR_FLIP).next_to(lab_3, DOWN, buff=0.05)
        lab_group_3 = VGroup(lab_3, sub_lab_3)
        
        inp_3.shift(UP*0.8)
        arr_3.next_to(inp_3, DOWN, buff=0.2)
        out_3.next_to(arr_3, DOWN, buff=0.2)
        lab_group_3.next_to(out_3, DOWN, buff=0.2)
        
        flip_group = VGroup(inp_3, out_3, arr_3, lab_group_3)
        # Issue 31 fix
        self.place_in_area(flip_group, 'E2', 'F5', scale_factor=0.8)
        
        self.play(Create(inp_3))
        self.play(GrowArrow(arr_3))
        self.play(Create(out_3), Write(lab_group_3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_ACC)
        )
        # Accordion effect: stretch/squeeze
        self.play(
            acc_1.animate.scale(1.3),
            acc_2.animate.scale(0.7),
            run_time=1.0,
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.play(
            Indicate(lab_group_1, color=COLOR_EXP),
            Indicate(lab_group_2, color=COLOR_COMP),
            Indicate(lab_group_3, color=COLOR_FLIP)
        )
        self.wait(2)
