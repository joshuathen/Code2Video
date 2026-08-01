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
        title = "Vector Addition: The Tip-to-Tail Method"
        lines = [
            "To combine vectors, we use the tip-to-tail method.",
            "Place the second vector's tail at the first's tip.",
            "The resulting vector connects the very start to the end.",
            "Algebraically, we simply add the corresponding x and y parts.",
            "This sum represents the total displacement of both movements."
        ]
        self.setup_layout(title, lines)

        # Define key grid positions for the animation
        # Origin at D2
        # Vector A: D2 -> D5 (Right 3)
        # Vector B: D2 -> B2 (Up 2)
        # Shifted Vector B: D5 -> B5 (Up 2)
        origin_pos = self.grid["D2"]
        a_tip_pos = self.grid["D5"]
        b_tip_pos = self.grid["B2"]
        b_shifted_tip_pos = self.grid["B5"]

        # === Animation for Lecture Line 1 ===
        # Draw Vector A in cyan #00FFFF and Vector B in magenta #FF00FF from the origin.
        self.lecture[0].set_color("#00FFFF")
        
        vec_a = Arrow(start=origin_pos, end=a_tip_pos, buff=0, color="#00FFFF", stroke_width=4)
        vec_b = Arrow(start=origin_pos, end=b_tip_pos, buff=0, color="#FF00FF", stroke_width=4)
        
        label_a = MathTex("A", color="#00FFFF")
        # Fix Issue 38: E3 -> E4 for better centering below Vector A
        self.place_at_grid(label_a, "E4", scale_factor=0.8)
        
        label_b = MathTex("B", color="#FF00FF")
        # Fix Issue 36: C1 -> C3 to avoid proximity to lecture notes.
        # Note: Using C3 instead of C5 to maintain proximity to the initial vector position.
        self.place_at_grid(label_b, "C3", scale_factor=0.8)

        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Shift the magenta #FF00FF Vector B so its tail meets Vector A's tip.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        
        shift_vector = a_tip_pos - origin_pos
        self.play(
            vec_b.animate.shift(shift_vector),
            label_b.animate.shift(shift_vector + RIGHT * 0.2), # Ends near C6, to the right of shifted vector
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Draw a white #FFFFFF dashed line from the origin to the new tip of B.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        dashed_res = DashedLine(start=origin_pos, end=b_shifted_tip_pos, color="#FFFFFF")
        self.play(Create(dashed_res))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Reveal the solid white #FFFFFF resultant vector and label it 'A + B'.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        vec_res = Arrow(start=origin_pos, end=b_shifted_tip_pos, buff=0, color="#FFFFFF", stroke_width=6)
        label_res = MathTex("A + B", color="#FFFFFF")
        # Fix Issue 37: C3 -> B2 to avoid overlap with the diagonal vector.
        self.place_at_grid(label_res, "B2", scale_factor=0.8)

        self.play(
            FadeOut(dashed_res),
            Create(vec_res),
            Write(label_res)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Highlight the 'A + B' label with a SurroundingRectangle in white #FFFFFF.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF")
        
        highlight_rect = SurroundingRectangle(label_res, color="#FFFFFF", buff=0.1)
        self.play(Create(highlight_rect))
        self.wait(2)
