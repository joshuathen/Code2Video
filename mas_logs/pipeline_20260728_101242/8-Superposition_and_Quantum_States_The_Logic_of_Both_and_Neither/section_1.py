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

class Section1Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "The Classical vs. Quantum Divide"
        lines = [
            "Classical bits are always either 0 or 1.",
            "Quantum bits exist in a mix of both states.",
            "Microscopic objects lack definite properties until observed."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Classical bit initially as 0 (White)
        c_bit_circle = Circle(radius=0.5, color=WHITE, fill_opacity=1)
        label_0 = Text("0", font_size=36, color=BLACK)
        bit_0_group = VGroup(c_bit_circle, label_0)
        # Issue 42: Move to B4 to avoid cluttering the left side
        self.place_at_grid(bit_0_group, "B4")
        
        self.play(Create(bit_0_group))
        self.wait(1)
        
        # Transition to 1 (Yellow)
        c_bit_circle_y = Circle(radius=0.5, color="#FFFF00", fill_opacity=1)
        label_1 = Text("1", font_size=36, color=BLACK)
        bit_1_group = VGroup(c_bit_circle_y, label_1)
        # Issue 42: Move to B4
        self.place_at_grid(bit_1_group, "B4")
        
        self.play(Transform(bit_0_group, bit_1_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2, Fade Line 1
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Quantum Cat (Qubit) as blurry overlay of white and yellow
        # Issue 25: Integrate cat asset
        cat_asset = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png").scale(0.5)
        
        q_circle_w = Circle(radius=0.6, color=WHITE, fill_opacity=0.3).set_stroke(opacity=0.2)
        q_circle_y = Circle(radius=0.6, color="#FFFF00", fill_opacity=0.3).set_stroke(opacity=0.2)
        # Shift slightly for blurry/overlap effect
        q_circle_y.shift(RIGHT*0.15 + UP*0.1)
        
        # Combine circles and cat icon into a Group
        q_cat = Group(q_circle_w, q_circle_y, cat_asset)
        # Issue 40: Position q_cat at C4 to avoid overlap with label
        self.place_at_grid(q_cat, "C4")
        
        qubit_label = Text("Quantum State", font_size=24, color=WHITE)
        # Issue 40: Position qubit_label at D4
        self.place_at_grid(qubit_label, "D4")
        
        self.play(FadeIn(q_cat), Write(qubit_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3, Fade Line 2
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#ADD8E6")
        )
        
        highlight_text = Text("No definite properties\nuntil observed", font_size=28, color="#ADD8E6")
        # Issue 41: Reposition highlight_text to area F3-F5 to avoid crowding
        self.place_in_area(highlight_text, "F3", "F5", scale_factor=0.8)
        
        self.play(Write(highlight_text))
        self.wait(2)
