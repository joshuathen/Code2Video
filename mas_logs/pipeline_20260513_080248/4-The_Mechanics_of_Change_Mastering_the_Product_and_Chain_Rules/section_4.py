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
        # Define lecture lines
        lecture_lines = [
            'Some problems require both the product and chain rules.',
            'Meet our jet-pack penguin with a complex altitude function.',
            'Identify the outer product of two distinct functions first.',
            'Apply the product rule while nesting the chain rule.',
            'Solve systematically by peeling back each mathematical layer.'
        ]
        
        self.setup_layout("Combined Application: The Jet-Pack Penguin", lecture_lines)

        # Helper to build powers without Tex
        def create_power(base_str, pwr_str, color=WHITE, font_size=24):
            base = Text(base_str, font_size=font_size, color=color)
            pwr = Text(pwr_str, font_size=font_size * 0.6, color=color)
            pwr.next_to(base.get_top(), RIGHT, buff=0.02).shift(UP * 0.05)
            return VGroup(base, pwr)

        # Colors
        RED_COLOR = "#FF0000"
        BLUE_COLOR = "#0000FF"
        HIGHLIGHT = "#FFFF00"
        WHITE_COLOR = "#FFFFFF"
        
        # Asset path
        PENGUIN_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/penguin.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT)
        rule_box = Rectangle(width=3, height=0.6, color=WHITE_COLOR)
        rule_text = Text("Product + Chain", font_size=20)
        rule_group = VGroup(rule_box, rule_text)
        # Issue 31 Fix: B2 to B5
        self.place_in_area(rule_group, "B2", "B5", scale_factor=1.0)
        self.play(FadeIn(rule_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Issue 21: Penguin introduces the function
        penguin = SVGMobject(PENGUIN_PATH)
        self.place_at_grid(penguin, "A6", scale_factor=0.5)

        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            self.lecture[1].animate.set_color(HIGHLIGHT),
            FadeOut(rule_group),
            FadeIn(penguin)
        )

        # H(t) = t^2 * sin(t^3)
        h_label = Text("H(t) = ", font_size=24)
        t_sq = create_power("t", "2", color=RED_COLOR)
        dot = Text(" · ", font_size=24)
        sin_part = Text("sin(", font_size=24, color=BLUE_COLOR)
        t_cubed = create_power("t", "3", color=BLUE_COLOR)
        paren_close = Text(")", font_size=24, color=BLUE_COLOR)
        
        func_vgroup = VGroup(h_label, t_sq, dot, sin_part, t_cubed, paren_close).arrange(RIGHT, buff=0.1)
        self.place_in_area(func_vgroup, "A2", "A5")
        
        self.play(Write(func_vgroup))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )

        # Outer Product Rule: (f*g)' = f'g + fg'
        prod_rule_template = Text("( f · g )' = f ' · g  +  f · g '", font_size=22, color=GREY_A)
        self.place_in_area(prod_rule_template, "B2", "B5")
        self.play(FadeIn(prod_rule_template))
        
        # Expanded form: (t^2)' * sin(t^3) + t^2 * (sin(t^3))'
        term1 = VGroup(Text("(", font_size=20), create_power("t", "2", color=RED_COLOR, font_size=20), Text(")'", font_size=20)).arrange(RIGHT, buff=0.05)
        term2 = VGroup(Text("sin(", font_size=20, color=BLUE_COLOR), create_power("t", "3", color=BLUE_COLOR, font_size=20), Text(")", font_size=20, color=BLUE_COLOR)).arrange(RIGHT, buff=0.05)
        plus = Text("+", font_size=20)
        term3 = create_power("t", "2", color=RED_COLOR, font_size=20)
        term4 = VGroup(Text("(sin(", font_size=20, color=BLUE_COLOR), create_power("t", "3", color=BLUE_COLOR, font_size=20), Text("))'", font_size=20)).arrange(RIGHT, buff=0.05)
        
        expansion = VGroup(term1, term2, plus, term3, term4).arrange(RIGHT, buff=0.2)
        self.place_in_area(expansion, "C2", "C5")
        
        self.play(TransformFromCopy(func_vgroup, expansion))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE_COLOR),
            self.lecture[3].animate.set_color(HIGHLIGHT)
        )

        # Focus on Differentiating
        deriv_step1 = Text("2t", font_size=22, color=RED_COLOR)
        deriv_step2 = VGroup(
            Text("cos(", font_size=22, color=BLUE_COLOR), 
            create_power("t", "3", color=BLUE_COLOR, font_size=22), 
            Text(") · ", font_size=22),
            create_power("3t", "2", color=HIGHLIGHT, font_size=22) # Chain rule part
        ).arrange(RIGHT, buff=0.05)

        full_deriv_step = VGroup(
            deriv_step1, 
            term2.copy(), 
            plus.copy(), 
            term3.copy(), 
            deriv_step2
        ).arrange(RIGHT, buff=0.2)
        
        # Issue 32 Fix: D1 to D6, scale 0.8
        self.place_in_area(full_deriv_step, "D1", "D6", scale_factor=0.8)
        
        # Show specific chain rule arrow
        arrow = Arrow(start=self.grid["C5"]+LEFT*0.5, end=self.grid["D5"]+RIGHT*0.5, color=HIGHLIGHT)
        chain_label = Text("Chain Rule", font_size=16, color=HIGHLIGHT)
        chain_label.next_to(arrow, UP)

        self.play(
            FadeIn(full_deriv_step),
            GrowArrow(arrow),
            FadeIn(chain_label)
        )
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        # Issue 21 celebration
        self.play(
            self.lecture[3].animate.set_color(WHITE_COLOR),
            self.lecture[4].animate.set_color(HIGHLIGHT),
            FadeOut(arrow),
            FadeOut(chain_label),
            penguin.animate.move_to(self.grid["E6"]).set_color(WHITE_COLOR)
        )

        # Final Result: H'(t) = 2t sin(t^3) + 3t^4 cos(t^3)
        h_prime = Text("H'(t) = ", font_size=26)
        final_term1 = VGroup(Text("2t sin(", font_size=24), create_power("t", "3", font_size=24), Text(")", font_size=24)).arrange(RIGHT, buff=0.05)
        final_plus = Text(" + ", font_size=24)
        # Simplify: t^2 * 3t^2 = 3t^4
        final_term2 = VGroup(create_power("3t", "4", font_size=24), Text(" cos(", font_size=24), create_power("t", "3", font_size=24), Text(")", font_size=24)).arrange(RIGHT, buff=0.05)
        
        final_formula = VGroup(h_prime, final_term1, final_plus, final_term2).arrange(RIGHT, buff=0.1)
        # Issue 33 Fix: E1 to E6, scale 0.9
        self.place_in_area(final_formula, "E1", "E6", scale_factor=0.9)
        
        self.play(Write(final_formula))
        self.play(
            Indicate(final_formula, color=WHITE_COLOR),
            Rotate(penguin, angle=2*PI) # Penguin celebration
        )
        self.wait(3)
