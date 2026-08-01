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
        # Data from storyboard/outline
        title = "Vector Addition: Combining Forces"
        lecture_lines = [
            "Adding vectors combines two different forces into one.",
            "Place the tail of the second vector at the first's tip.",
            "The 'resultant' vector connects the starting point to the end.",
            "Algebraically, we simply add the corresponding coordinates together.",
            "This sum represents the total displacement from both movements."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from Storyboard
        color_a = "#00FFFF"  # Cyan for Vector A
        color_b = "#FFA500"  # Orange for Vector B
        color_c = "#FF1493"  # DeepPink for Resultant Vector C
        color_math = "#F5DEB3" # Wheat for Algebra/Labels

        # === Animation for Lecture Line 1 ===
        # Draw Vector A (horizontal) and Vector B (vertical) starting at origin.
        self.play(self.lecture[0].animate.set_color(color_a))
        
        # Origin defined at E2
        origin = self.grid["E2"]
        
        # Vector A: [3, 0] -> E2 to E5
        vector_a = Arrow(origin, self.grid["E5"], buff=0, color=color_a)
        label_a = MathTex(r"\vec{A} = [3, 0]", color=color_a, font_size=24).next_to(vector_a, DOWN, buff=0.1)
        
        # Vector B: [0, 4] -> E2 to A2 (Initially both start at origin)
        vector_b = Arrow(origin, self.grid["A2"], buff=0, color=color_b)
        label_b = MathTex(r"\vec{B} = [0, 4]", color=color_b, font_size=24).next_to(vector_b, LEFT, buff=0.1)
        
        self.play(GrowArrow(vector_a), Write(label_a))
        self.play(GrowArrow(vector_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shift Vector B so its tail meets Vector A's tip (Tip-to-Tail).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_b)
        )
        
        # Move Vector B's tail from E2 to E5. New tip will be A5.
        target_b_pos = self.grid["E5"] - self.grid["E2"]
        
        self.play(
            vector_b.animate.shift(target_b_pos),
            label_b.animate.shift(target_b_pos)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw Resultant Vector C from origin to Vector B's tip.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_c)
        )
        
        # Resultant Vector C: E2 to A5
        vector_c = Arrow(self.grid["E2"], self.grid["A5"], buff=0, color=color_c)
        label_c = MathTex(r"\vec{C} = \vec{A} + \vec{B}", color=color_c, font_size=28)
        
        # Fix for Issue 22: Improved positioning for label_c
        self.place_in_area(label_c, 'B2', 'B4', scale_factor=0.8)
        
        self.play(GrowArrow(vector_c))
        self.play(Write(label_c))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show coordinates [3, 0] + [0, 4] = [3, 4] in #F5DEB3.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(color_math)
        )
        
        # Algebraic sum formula
        sum_formula = MathTex(r"[3, 0] + [0, 4] = [3, 4]", color=color_math, font_size=32)
        
        # Fix for Issue 23: Centered positioning for sum_formula
        self.place_in_area(sum_formula, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(sum_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse Vector C and label it "Net Displacement [3, 4]" in #FF1493.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(color_c)
        )
        
        label_displacement = Text("Net Displacement [3, 4]", color=color_c, font_size=20)
        self.place_at_grid(label_displacement, "A4", scale_factor=0.8)
        label_displacement.shift(UP * 0.4)
        
        self.play(Write(label_displacement))
        # Pulse animation using a ValueTracker for stroke width
        self.play(vector_c.animate.set_stroke(width=10), run_time=0.5)
        self.play(vector_c.animate.set_stroke(width=4), run_time=0.5)
        self.wait(2)
