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
        title_text = "Vector Addition: The 'Head-to-Tail' Rule"
        lecture_lines = [
            "To add vectors, place one tail at another's head.",
            "The sum is the shortcut from start to finish.",
            "This 'head-to-tail' rule combines two different movements."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define Colors
        color_a = "#FF0000" # Red
        color_b = "#00FF00" # Green
        color_c = "#FF00FF" # Magenta

        # Define Mobjects
        # === Stage 1: Vector A and B ===
        # Vector A: starts at C2, ends at C5
        v_a = Arrow(start=self.grid["C2"], end=self.grid["C5"], color=color_a, buff=0)
        label_a = MathTex(r"\vec{A}", color=color_a)
        # Issue 30: Moved label_a to E3 to avoid horizontal path overlap
        self.place_at_grid(label_a, "E3", scale_factor=0.8)

        # B initially separate (Length 2 North)
        v_b_start_init = self.grid["F2"]
        v_b_end_init = self.grid["D2"]
        v_b = Arrow(start=v_b_start_init, end=v_b_end_init, color=color_b, buff=0)
        label_b = MathTex(r"\vec{B}", color=color_b)
        # Issue 31: Placed label_b at C5 as requested for initial grid placement
        self.place_at_grid(label_b, "C5", scale_factor=0.8)

        # Target for B (Head of A to A5)
        target_v_b_start = self.grid["C5"]
        v_b_shift = target_v_b_start - v_b_start_init

        # === Stage 2: Vector C (Resultant) ===
        v_c = Arrow(start=self.grid["C2"], end=self.grid["A5"], color=color_c, buff=0)
        label_c = MathTex(r"\vec{C}", color=color_c)
        # Issue 32: Moved label_c to B2 to avoid obstruction
        self.place_at_grid(label_c, "B2", scale_factor=0.8)

        # === Stage 5: Equation ===
        eq_a = MathTex(r"\vec{A}", color=color_a)
        plus = MathTex("+", color=WHITE)
        eq_b = MathTex(r"\vec{B}", color=color_b)
        equals = MathTex("=", color=WHITE)
        eq_c = MathTex(r"\vec{C}", color=color_c)
        equation = VGroup(eq_a, plus, eq_b, equals, eq_c).arrange(RIGHT, buff=0.2)
        self.place_in_area(equation, "F2", "F5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # "To add vectors, place one tail at another's head."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(v_a), Write(label_a))
        self.wait(0.5)
        self.play(Create(v_b))
        self.wait(0.5)
        # Place tail of B at head of A
        self.play(
            v_b.animate.shift(v_b_shift),
            run_time=2
        )
        # Write label_b at its grid position C5 (tail of B's final position)
        self.play(Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The sum is the shortcut from start to finish."
        self.play(self.lecture[1].animate.set_color(color_c))
        self.play(Create(v_c), Write(label_c))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This 'head-to-tail' rule combines two different movements."
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Pulse Vector C while A and B fade slightly
        self.play(
            v_a.animate.set_stroke(opacity=0.3),
            label_a.animate.set_fill(opacity=0.3),
            v_b.animate.set_stroke(opacity=0.3),
            label_b.animate.set_fill(opacity=0.3),
            run_time=1
        )
        self.play(v_c.animate.set_stroke(width=10), run_time=0.5)
        self.play(v_c.animate.set_stroke(width=4), run_time=0.5)
        self.wait(0.5)
        
        # Issue 20: Integrate SVG Asset for Dot
        dot_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dot.svg").scale(0.15)
        dot_asset.set_color(WHITE)
        dot_asset.move_to(self.grid["C2"])
        
        self.play(FadeIn(dot_asset))
        # Move East along A
        self.play(dot_asset.animate.move_to(self.grid["C5"]), run_time=1, rate_func=linear)
        # Move North along B
        self.play(dot_asset.animate.move_to(self.grid["A5"]), run_time=1, rate_func=linear)
        self.wait(0.5)
        # Move along diagonal C
        self.play(dot_asset.animate.move_to(self.grid["C2"]), run_time=0.1) # Teleport back
        self.play(dot_asset.animate.move_to(self.grid["A5"]), run_time=1.5, rate_func=linear)
        self.play(FadeOut(dot_asset))

        # Show the vector addition equation
        self.play(Write(equation))
        self.wait(2)
