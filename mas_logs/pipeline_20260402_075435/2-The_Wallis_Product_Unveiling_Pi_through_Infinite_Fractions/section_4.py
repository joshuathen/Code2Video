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
        # Initialize layout
        self.setup_layout(
            "Even vs. Odd: The Convergence Logic", 
            [
                "Even and odd terms form two distinct sequences.",
                "We compare terms from both sequences side by side.",
                "Their ratios provide the key to convergence.",
                "As n grows, the ratio approaches exactly one.",
                "This squeeze effect forces the sequences to merge."
            ]
        )
        
        # Colors based on visual flow requirements
        COLOR_EVEN_ODD = "#58C4DD"
        COLOR_RATIO = "#F4D03F"
        COLOR_INEQUALITY = "#FF7043"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_EVEN_ODD))
        
        # Headers for the two sequences
        even_header = Text("Even", font_size=24, color=WHITE)
        odd_header = Text("Odd", font_size=24, color=WHITE)
        self.place_at_grid(even_header, "A1", scale_factor=1.0)
        self.place_at_grid(odd_header, "A6", scale_factor=1.0)
        
        # Column separator line
        sep_start = (self.grid["A3"] + self.grid["A4"]) / 2 + UP * 0.4
        sep_end = (self.grid["F3"] + self.grid["F4"]) / 2 + DOWN * 0.4
        separator = Line(sep_start, sep_end, color=WHITE)
        
        self.play(
            FadeIn(even_header), 
            FadeIn(odd_header), 
            Create(separator)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_EVEN_ODD))
        
        # Sequences (Using Text instead of MathTex to avoid LaTeX dependency)
        even_terms = VGroup(
            Text("I_0", color=COLOR_EVEN_ODD, font_size=32),
            Text("I_2", color=COLOR_EVEN_ODD, font_size=32),
            Text("I_4", color=COLOR_EVEN_ODD, font_size=32)
        ).arrange(DOWN, buff=0.6)
        
        odd_terms = VGroup(
            Text("I_1", color=COLOR_EVEN_ODD, font_size=32),
            Text("I_3", color=COLOR_EVEN_ODD, font_size=32),
            Text("I_5", color=COLOR_EVEN_ODD, font_size=32)
        ).arrange(DOWN, buff=0.6)
        
        # Positioning according to Issue 45 fix
        self.place_at_grid(even_terms, "B1", scale_factor=0.9)
        self.place_at_grid(odd_terms, "B6", scale_factor=0.9)
        
        self.play(FadeIn(even_terms), FadeIn(odd_terms))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_RATIO))
        
        # Ratio representation - positioning according to Issue 43 fix
        ratio_tex = Text("Ratio I(2n) / I(2n-1) -> 1", color=COLOR_RATIO, font_size=28)
        self.place_in_area(ratio_tex, "B3", "C4", scale_factor=1.0)
        self.play(Write(ratio_tex))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_RATIO))
        
        # Number line for visual ratio convergence
        num_line = NumberLine(
            x_range=[0.5, 1.5, 0.25],
            length=4.5,
            color=WHITE,
            include_numbers=True,
            font_size=18,
            numbers_to_include=[0.5, 1.0, 1.5],
            label_constructor=Text
        )
        self.place_in_area(num_line, "F1", "F6", scale_factor=1.0)
        
        # Ratio movement indicator using persistent mobjects
        ratio_tracker = ValueTracker(1.4)
        pointer = Arrow(UP, DOWN, color=COLOR_RATIO, buff=0).scale(0.4)
        pointer.add_updater(lambda m: m.next_to(num_line.n2p(ratio_tracker.get_value()), UP, buff=0.1))
        
        val_label = DecimalNumber(
            1.4, 
            num_decimal_places=2, 
            color=COLOR_RATIO, 
            font_size=20,
            mob_class=Text
        )
        val_label.add_updater(lambda m: m.set_value(ratio_tracker.get_value()).next_to(pointer, UP, buff=0.1))
        
        self.play(Create(num_line))
        self.add(pointer, val_label)
        self.play(ratio_tracker.animate.set_value(1.0), run_time=3, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_INEQUALITY))
        
        # Monotonicity inequality - positioning according to Issue 44 fix
        inequality = Text("I(2n) > I(2n+1) > I(2n+2)", color=COLOR_INEQUALITY, font_size=32)
        self.place_in_area(inequality, "D1", "D6", scale_factor=0.9)
        
        self.play(Write(inequality))
        # Visual flash for emphasis
        self.play(Flash(inequality, color=COLOR_INEQUALITY, line_length=0.4))
        self.play(
            inequality.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=0.4
        )
        
        self.wait(2)
