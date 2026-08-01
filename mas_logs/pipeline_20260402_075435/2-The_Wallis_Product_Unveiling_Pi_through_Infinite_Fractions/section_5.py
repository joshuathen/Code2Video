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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Final Product: Piecing it Together"
        lines = [
            "Combine these sequences to isolate the value of Pi.",
            "Fractions from both sequences begin to intertwine.",
            "Rearranging these terms creates a chain of ratios.",
            "We arrive at the elegant Wallis Product formula.",
            "The numeric value converges slowly to Pi halves."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        even_product = Text(
            "I(2n) = [(2n-1)/2n] · [(2n-3)/(2n-2)] · ... · (1/2) · (π/2)",
            color=WHITE, font_size=24
        )
        self.place_in_area(even_product, 'A2', 'A6', scale_factor=0.7)
        
        odd_product = Text(
            "I(2n+1) = [2n/(2n+1)] · [(2n-2)/(2n-1)] · ... · (2/3)",
            color="#58C4DD", font_size=24
        )
        self.place_in_area(odd_product, 'B2', 'B6', scale_factor=0.7)
        
        self.play(Write(even_product), run_time=1.5)
        self.play(Write(odd_product), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#58C4DD")
        
        ratio_formula = Text(
            "π/2 = [I(2n)/I(2n+1)] · (2/1 · 2/3 · 4/3 · ... · 2n/(2n+1))",
            color="#F4D03F", font_size=22
        )
        self.place_in_area(ratio_formula, "C1", "C6", scale_factor=0.6)
        self.play(FadeIn(ratio_formula, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#F4D03F")
        
        self.play(FadeOut(even_product), FadeOut(odd_product), FadeOut(ratio_formula))
        
        try:
            block_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/blocks.svg")
        except:
            block_svg = Square().scale(0.5)

        block_svg.set_color("#58C4DD").set_stroke(width=1)
        
        fractions_list = ["2/1", "2/3", "4/3", "4/5", "6/5", "6/7"]
        lego_group = VGroup()
        for frac_str in fractions_list:
            brick_icon = block_svg.copy()
            text_mob = Text(frac_str, color=WHITE, font_size=24).move_to(brick_icon.get_center())
            lego_group.add(VGroup(brick_icon, text_mob))
        
        lego_group.arrange(RIGHT, buff=0.1)
        self.place_in_area(lego_group, 'C1', 'C6', scale_factor=0.65)
        
        self.play(LaggedStart(*[FadeIn(brick, shift=UP) for brick in lego_group], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        
        wallis_formula = Text(
            "π/2 = ∏ (n=1 to ∞) [ (2n/(2n-1)) · (2n/(2n+1)) ]",
            color=WHITE, font_size=28
        )
        self.place_in_area(wallis_formula, "D1", "D6", scale_factor=0.8)
        
        frame = SurroundingRectangle(wallis_formula, color=WHITE, buff=0.2, stroke_width=2)
        
        self.play(Write(wallis_formula))
        self.play(Create(frame))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F4D03F")
        
        def calculate_wallis(k):
            val = 1.0
            for i in range(1, k + 1):
                if i % 2 == 1:
                    val *= (i + 1) / i
                else:
                    val *= i / (i + 1)
            return val

        term_tracker = ValueTracker(1)
        
        counter = DecimalNumber(
            calculate_wallis(1), 
            num_decimal_places=4, 
            color="#F4D03F",
            font_size=40,
            mob_class=Text
        )
        approx_label = Text("≈", color="#F4D03F", font_size=40)
        approx_group = VGroup(approx_label, counter).arrange(RIGHT, buff=0.2)
        
        self.place_at_grid(approx_group, 'E4', scale_factor=1.1)
        
        counter.add_updater(lambda c: c.set_value(calculate_wallis(int(term_tracker.get_value()))))
        
        self.play(FadeIn(approx_group))
        self.play(term_tracker.animate.set_value(300), run_time=4, rate_func=linear)
        
        counter.remove_updater(counter.updaters[0])
        counter.set_value(1.5708)
        self.play(Indicate(counter))
        self.wait(2)
