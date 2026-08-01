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
    def construct(self):
        # Title and Lecture Lines
        title = "The Pattern Emerges: Scaling the Mass"
        lecture_lines = [
            'With equal masses, we count exactly three collisions.',
            'At a hundred-to-one ratio, thirty-one collisions occur.',
            'Increasing the mass further reveals a startling pattern.',
            'The count perfectly matches the digits of Pi.',
            'This mechanical system actually computes the constant Pi.'
        ]
        
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Simulation components for M=1, m=1
        equal_mass_sim = VGroup()
        wall1 = Line(UP, DOWN).scale(0.8)
        floor1 = Line(LEFT, RIGHT).scale(2.5).next_to(wall1.get_bottom(), RIGHT, buff=0)
        block_M1 = Square(side_length=0.4, fill_opacity=1, color=BLUE).next_to(floor1, UP, buff=0).shift(RIGHT*1.5)
        block_m1 = Square(side_length=0.4, fill_opacity=1, color=RED).next_to(floor1, UP, buff=0).shift(RIGHT*0.5)
        label_M1 = Text("M=1", font_size=18).next_to(block_M1, UP, buff=0.1)
        label_m1 = Text("m=1", font_size=18).next_to(block_m1, UP, buff=0.1)
        
        # Fixed FileNotFoundError by using mob_class=Text for Integer
        count_val1 = Integer(0, color=WHITE, mob_class=Text).scale(0.8)
        count_label1 = VGroup(Text("Collisions: ", font_size=18), count_val1).arrange(RIGHT, buff=0.1)
        count_label1.next_to(floor1, UP, buff=1.0)
        
        equal_mass_sim.add(wall1, floor1, block_M1, block_m1, label_M1, label_m1, count_label1)
        self.place_in_area(equal_mass_sim, 'A1', 'B6', scale_factor=0.6)
        
        self.play(FadeIn(equal_mass_sim))
        
        self.play(block_M1.animate.shift(LEFT*0.5), label_M1.animate.shift(LEFT*0.5), run_time=0.5)
        count_val1.set_value(1) # M hits m
        self.play(block_m1.animate.shift(LEFT*0.4), label_m1.animate.shift(LEFT*0.4), run_time=0.4)
        count_val1.set_value(2) # m hits wall
        self.play(block_m1.animate.shift(RIGHT*0.4), label_m1.animate.shift(RIGHT*0.4), run_time=0.4)
        count_val1.set_value(3) # m hits M
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Simulation components for M=100, m=1
        ratio_100_sim = VGroup()
        wall2 = Line(UP, DOWN).scale(0.8)
        floor2 = Line(LEFT, RIGHT).scale(2.5).next_to(wall2.get_bottom(), RIGHT, buff=0)
        block_M2 = Square(side_length=0.6, fill_opacity=1, color=BLUE).next_to(floor2, UP, buff=0).shift(RIGHT*1.5)
        block_m2 = Square(side_length=0.3, fill_opacity=1, color=RED).next_to(floor2, UP, buff=0).shift(RIGHT*0.5)
        label_M2 = Text("M=100", font_size=18).next_to(block_M2, UP, buff=0.1)
        label_m2 = Text("m=1", font_size=18).next_to(block_m2, UP, buff=0.1)
        
        count_val2 = Integer(0, color=WHITE, mob_class=Text).scale(0.8)
        count_label2 = VGroup(Text("Collisions: ", font_size=18), count_val2).arrange(RIGHT, buff=0.1)
        count_label2.next_to(floor2, UP, buff=1.0)
        
        ratio_100_sim.add(wall2, floor2, block_M2, block_m2, label_M2, label_m2, count_label2)
        self.place_in_area(ratio_100_sim, 'C1', 'D6', scale_factor=0.6)
        
        self.play(FadeIn(ratio_100_sim))
        self.play(count_val2.animate.set_value(31), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Simulation components for M=10,000, m=1
        pi_calculation_display = VGroup()
        wall3 = Line(UP, DOWN).scale(0.8)
        floor3 = Line(LEFT, RIGHT).scale(2.5).next_to(wall3.get_bottom(), RIGHT, buff=0)
        block_M3 = Square(side_length=0.8, fill_opacity=1, color=BLUE).next_to(floor3, UP, buff=0).shift(RIGHT*1.5)
        block_m3 = Square(side_length=0.3, fill_opacity=1, color=RED).next_to(floor3, UP, buff=0).shift(RIGHT*0.5)
        label_M3 = Text("M=10,000", font_size=18).next_to(block_M3, UP, buff=0.1)
        label_m3 = Text("m=1", font_size=18).next_to(block_m3, UP, buff=0.1)
        
        count_val3 = Integer(0, color=WHITE, mob_class=Text).scale(1.0)
        count_label3 = VGroup(Text("Collisions: ", font_size=20), count_val3).arrange(RIGHT, buff=0.1)
        count_label3.next_to(floor3, UP, buff=1.0)
        
        pi_calculation_display.add(wall3, floor3, block_M3, block_m3, label_M3, label_m3, count_label3)
        self.place_in_area(pi_calculation_display, 'E1', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(pi_calculation_display))
        self.play(count_val3.animate.set_value(314), run_time=2.0, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(PINK)
        
        self.play(count_val3.animate.set_color("#FF00FF").scale(1.5))
        self.play(count_val3.animate.scale(1/1.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PINK)
        
        # Avoid LaTeX by using Text instead of MathTex
        pi_symbol = Text("π ≈ 3.141...", color="#FF00FF").scale(1.2)
        pi_symbol.move_to(count_label3.get_center())
        
        ratio_1M_label = Text("Ratio 1,000,000 : 1 -> 3141", font_size=20, color="#FF00FF")
        ratio_1M_label.next_to(pi_symbol, DOWN, buff=0.5)
        
        self.play(ReplacementTransform(count_label3, pi_symbol))
        self.play(Write(ratio_1M_label))
        self.wait(2)
