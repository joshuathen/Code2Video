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
        # Title and Lecture Lines from storyboard
        title_text = "Defining the Zeta Function"
        lecture_lines = [
            "We define the Riemann Zeta function using these sums.",
            "Input any number \"s\" into this function machine.",
            "The output is the sum of reciprocal powers.",
            "For s equals four, the sum is exactly pi-fourth over ninety.",
            "Each point creates a curve in our mathematical landscape."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Summation formula using Text
        sum_formula = Text("Σ 1/nˢ", color=WHITE, font_size=36)
        self.place_in_area(sum_formula, "B3", "C4", scale_factor=1.2)
        self.play(Write(sum_formula))
        
        zeta_label = Text("ζ(s)", color="#FFFFFF", font_size=42) # Bold white per storyboard
        self.place_in_area(zeta_label, "B3", "C4", scale_factor=1.5)
        
        # Replace the summation symbol with ζ(s)
        self.play(FadeOut(sum_formula), FadeIn(zeta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00") # Matching green input
        
        # Asset: machine.svg [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/machine.svg]
        # Issue 30: Positioned at D3-D4
        machine = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/machine.svg")
        machine.set_color(GREY_B)
        self.place_in_area(machine, "D3", "D4", scale_factor=0.8)
        
        input_s = Text("s", color="#00FF00", font_size=36)
        self.place_at_grid(input_s, "C2") # Issue 28: moved from C1 to C2
        
        self.play(
            FadeIn(machine), 
            zeta_label.animate.scale(0.6).move_to(self.grid["C3"])
        )
        # Input value s moves into the machine
        self.play(input_s.animate.move_to(self.grid["D3"]), run_time=1)
        self.play(FadeOut(input_s, shift=DOWN*0.3), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Output emerges from the machine
        output_sum = Text("Σ 1/nˢ", color=WHITE, font_size=36)
        self.place_at_grid(output_sum, "D5") # Issue 29: moved from D6 to D5
        
        self.play(FadeIn(output_sum, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFD700") # Matching gold output
        
        # Specific input s=4
        input_4 = Text("4", color="#00FF00", font_size=36)
        self.place_at_grid(input_4, "C2") # Issue 28: moved from C1 to C2
        
        # Output π⁴/90 (#FFD700) emerges with a glow effect
        output_val = Text("π⁴/90", color="#FFD700", font_size=36)
        self.place_at_grid(output_val, "D5") # Issue 29: moved from D6 to D5
        
        self.play(FadeOut(output_sum))
        self.play(input_4.animate.move_to(self.grid["D3"]), run_time=1)
        self.play(FadeOut(input_4, shift=DOWN*0.3), run_time=0.5)
        
        # Glow effect
        glow = output_val.copy().set_stroke(width=10, opacity=0.4).set_color("#FFD700")
        self.play(FadeIn(output_val, shift=RIGHT), FadeIn(glow, shift=RIGHT))
        self.play(glow.animate.scale(1.5).set_opacity(0), run_time=1.2)
        self.remove(glow)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Small axes for the landscape. Issue 30: positioned at E2-F5
        axes = Axes(
            x_range=[1, 6, 1],
            y_range=[1, 2, 0.5],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        self.place_in_area(axes, "E2", "F5", scale_factor=0.8)
        
        # Point at s=4, zeta(4) approx 1.0823
        point_pos = axes.c2p(4, 1.0823)
        dot = Dot(point_pos, color="#FFD700")
        dot_label = Text("ζ(4)", color="#FFD700", font_size=20).next_to(dot, UP, buff=0.1)
        
        self.play(Create(axes))
        self.play(TransformFromCopy(output_val, dot), FadeIn(dot_label))
        
        # Approximation of the zeta function curve
        zeta_curve = axes.plot(lambda x: 1 + (1/2)**x + (1/3)**x + (1/4)**x, x_range=[1.2, 5.5], color=WHITE)
        self.play(Create(zeta_curve), run_time=2)
        self.wait(2)
