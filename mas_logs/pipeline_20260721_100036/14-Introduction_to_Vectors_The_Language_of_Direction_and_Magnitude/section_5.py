from manim import *

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
        self.setup_layout("Scalar Multiplication: Scaling the Force", [
            "Multiplying a vector by a scalar changes its length.",
            "Positive scalars stretch or shrink the arrow's size.",
            "Negative scalars flip the vector's direction completely."
        ])
        
        # Colors
        ORANGE_V = "#FFA500"
        RED_V = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Text: Multiplying a vector by a scalar changes its length.
        # Anim: An orange #FFA500 vector [1, 2] appears, then stretches to [2, 4].
        
        self.lecture[0].set_color(ORANGE_V)
        
        # Using a coordinate-aligned set of points on the 6x6 grid.
        # Origin: D3
        # v1: [1, 1] -> C4
        # v2: [2, 2] -> B5 (Represents stretching to [2, 4] concept)
        
        origin = self.grid['D3']
        v1_end = self.grid['C4']
        v2_end = self.grid['B5']
        
        vector = Arrow(origin, v1_end, buff=0, color=ORANGE_V)
        v_label = MathTex(r"\vec{v}", color=ORANGE_V, font_size=24)
        # Fix Issue 33: v_label moved from C5 to B5 to avoid overlap with v1's tip (C4)
        self.place_at_grid(v_label, 'B5', scale_factor=0.8)

        self.play(Create(vector), Write(v_label))
        self.wait(0.5)
        
        self.play(
            vector.animate.put_start_and_end_on(origin, v2_end),
            v_label.animate.move_to(self.grid['B6']) # Move label to B6 as vector extends to B5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Text: Positive scalars stretch or shrink the arrow's size.
        # Anim: The label '2 * v' appears, and the vector flashes in orange #FFA500.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ORANGE_V)
        
        label_2v = MathTex(r"2 \cdot \vec{v}", color=ORANGE_V, font_size=24)
        # Fix Issue 34: Scale label_2v to 0.8 for consistency
        self.place_at_grid(label_2v, 'B6', scale_factor=0.8) 
        
        self.play(FadeOut(v_label), FadeIn(label_2v))
        self.play(Indicate(vector, color=ORANGE_V))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Text: Negative scalars flip the vector's direction completely.
        # Anim: The vector rotates 180 degrees and points the opposite way, changing to red #FF0000.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED_V)
        
        # End point for -2v: D3 to F1 (2 units down, 2 left)
        v_neg_end = self.grid['F1']
        label_neg_2v = MathTex(r"-2 \cdot \vec{v}", color=RED_V, font_size=24)
        # Fix Issue 35: label_neg_2v moved from F2 to E2 to avoid overlap with tip at F1
        self.place_at_grid(label_neg_2v, 'E2', scale_factor=0.8) 

        self.play(
            vector.animate.put_start_and_end_on(origin, v_neg_end).set_color(RED_V),
            FadeOut(label_2v),
            FadeIn(label_neg_2v)
        )
        self.play(Indicate(vector, color=RED_V))
        self.wait(2)
