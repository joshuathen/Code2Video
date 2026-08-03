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
        self.setup_layout("Prerequisites: Vectors as Directions", [
            "Vectors represent movements or directions in space.",
            "We can scale vectors to stretch or shrink them.",
            "Adding scaled vectors creates a linear combination.",
            "For example, three steps East and two steps North.",
            "This operation combines basic movements into new paths."
        ])

        # Colors
        color_a = "#FFD700"  # Gold/Yellow
        color_b = "#00BFFF"  # Deep Sky Blue
        color_res = "#ADFF2F" # GreenYellow

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_a)
        
        # Vector A: East (1 step) starting at D2
        origin = self.grid["D2"]
        target_a = self.grid["D3"]
        vector_a = Arrow(origin, target_a, buff=0, color=color_a)
        label_a = MathTex("A", color=color_a).scale(0.8)
        self.place_at_grid(label_a, "E3") # Positioned within 1 unit
        
        self.play(Create(vector_a), Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_a)
        
        # Stretch to double length (k=2)
        target_2a = self.grid["D4"]
        label_ka = MathTex("k \\cdot A", color=color_a).scale(0.8)
        self.place_at_grid(label_ka, "E4")
        
        self.play(
            vector_a.animate.put_start_and_end_on(origin, target_2a),
            Transform(label_a, label_ka)
        )
        self.wait(0.5)
        
        # Flip West (k=-1)
        target_minus_a = self.grid["D1"] # West from D2
        self.play(
            vector_a.animate.put_start_and_end_on(origin, target_minus_a)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_res)
        
        # Cleanup first part
        self.play(FadeOut(vector_a, label_a))
        
        # Show A (East) and B (North)
        vector_a_new = Arrow(origin, self.grid["D3"], buff=0, color=color_a)
        label_a_new = MathTex("A", color=color_a).scale(0.7)
        self.place_at_grid(label_a_new, "E3")
        
        vector_b = Arrow(origin, self.grid["C2"], buff=0, color=color_b)
        label_b = MathTex("B", color=color_b).scale(0.7)
        self.place_at_grid(label_b, "C1")
        
        self.play(Create(vector_a_new), Create(vector_b), Write(label_a_new), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(color_res)
        
        # Scale to 3A and 2B
        target_3a = self.grid["D5"]
        target_2b = self.grid["B2"]
        
        label_3a = MathTex("3A", color=color_a).scale(0.7)
        # FIX ISSUE 19: Move label_3a to E3
        self.place_at_grid(label_3a, "E3")
        
        label_2b_scaled = MathTex("2B", color=color_b).scale(0.7)
        # FIX ISSUE 18: Move label_2b to B6 as requested
        self.place_at_grid(label_2b_scaled, "B6")
        
        self.play(
            vector_a_new.animate.put_start_and_end_on(origin, target_3a),
            vector_b.animate.put_start_and_end_on(origin, target_2b),
            Transform(label_a_new, label_3a),
            Transform(label_b, label_2b_scaled)
        )
        self.wait(1)
        
        # Tip-to-tail: Move 2B to end of 3A
        target_tip_to_tail = self.grid["B5"]
        
        self.play(
            vector_b.animate.put_start_and_end_on(target_3a, target_tip_to_tail)
        )
        self.wait(1)

        # Resultant vector
        resultant = Arrow(origin, target_tip_to_tail, buff=0, color=color_res)
        label_res = MathTex("3A + 2B", color=color_res).scale(0.8)
        # FIX ISSUE 20: Move label_res to A5
        self.place_at_grid(label_res, "A5")
        
        self.play(Create(resultant), Write(label_res))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(color_res)
        
        # Final highlight
        self.play(Indicate(resultant))
        self.wait(2)
