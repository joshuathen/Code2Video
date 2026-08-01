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
        # Fetching Title and Lecture Lines from storyboard
        title_text = "Defining the Zeta Function: The Tuning Knob"
        lecture_lines = [
            "We define the Zeta function using the variable s.",
            "Think of s as a dial that tunes the sum.",
            "As Pete turns the dial, the sum’s value changes.",
            "Inputting two yields the famous Basel Problem result.",
            "This function maps numbers to specific points in space."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Define Colors for each step to match lecture lines with animation elements
        COLOR_1 = WHITE
        COLOR_2 = "#FF8C00" # Orange for Pete and Dial
        COLOR_3 = "#FF8C00" # Still Orange for interaction
        COLOR_4 = "#00BFFF" # Deep Sky Blue for Basel result
        COLOR_5 = "#FFFF00" # Yellow for expanded mapping

        # === Animation for Lecture Line 1 ===
        # Using Text with Unicode instead of MathTex to avoid FileNotFoundError: 'latex'
        self.lecture[0].set_color(COLOR_1)
        zeta_formula = Text("ζ(s) = Σ 1/nˢ", color=COLOR_1)
        # Layout Refinement: Zeta formula in area 'A1' to 'B3' (scale 0.8)
        self.place_in_area(zeta_formula, "A1", "B3", scale_factor=0.8)
        
        self.play(Write(zeta_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce a dial 's'
        self.lecture[1].set_color(COLOR_2)
        
        # Pete character at 'D1' (scale 0.6)
        pete_circle = Circle(radius=0.4, color="#00FF00")
        pete_label = Text("Pete", font_size=18, color="#00FF00")
        pete = VGroup(pete_circle, pete_label)
        pete_label.next_to(pete_circle, DOWN, buff=0.1)
        self.place_at_grid(pete, "D1", scale_factor=0.6)
        
        # Dial at 'D2' (scale 0.7)
        dial_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dial.svg")
        dial_svg.set_color(COLOR_2)
        self.place_at_grid(dial_svg, "D2", scale_factor=0.7)
        
        dial_label = Text("s", color=COLOR_2).next_to(dial_svg, UP, buff=0.1)
        dial_group = VGroup(dial_svg, dial_label)
        
        self.play(FadeIn(pete), FadeIn(dial_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_3)
        
        # Sum expansion in 'C1' to 'C6' (scale 0.7)
        sum_expansion = Text("= 1 + 1/2ˢ + 1/3ˢ + 1/4ˢ + ...", color=COLOR_1)
        self.place_in_area(sum_expansion, "C1", "C6", scale_factor=0.7)
        
        # Pete turns the dial
        self.play(
            FadeIn(sum_expansion),
            Rotate(dial_svg, angle=PI/4, about_point=dial_svg.get_center())
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_4)
        
        # Transition sum to s=2
        sum_s2 = Text("= 1 + 1/2² + 1/3² + 1/4² + ...", color=COLOR_4)
        self.place_in_area(sum_s2, "C1", "C6", scale_factor=0.7)
        
        # Basel result at 'E4' (scale 1.1)
        basel_result = Text("→ π²/6", color=COLOR_4)
        self.place_at_grid(basel_result, "E4", scale_factor=1.1)
        
        self.play(
            Rotate(dial_svg, angle=PI/8, about_point=dial_svg.get_center()),
            Transform(sum_expansion, sum_s2),
            FadeIn(basel_result)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_5)
        
        # Transition sum to s=4
        sum_s4 = Text("= 1 + 1/2⁴ + 1/3⁴ + 1/4⁴ + ...", color=COLOR_5)
        self.place_in_area(sum_s4, "C1", "C6", scale_factor=0.7)
        
        # Update Basel result to s=4 value
        basel_result_4 = Text("→ π⁴/90", color=COLOR_5)
        self.place_at_grid(basel_result_4, "E4", scale_factor=1.1)
        
        # Scale the formula slightly to emphasize its role as a machine
        self.play(
            Rotate(dial_svg, angle=PI/8, about_point=dial_svg.get_center()),
            Transform(sum_expansion, sum_s4),
            Transform(basel_result, basel_result_4),
            zeta_formula.animate.scale(1.1)
        )
        self.wait(2)
