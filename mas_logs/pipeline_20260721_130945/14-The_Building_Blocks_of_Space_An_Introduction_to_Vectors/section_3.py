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

class Section3Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines from Storyboard
        title_text = "Vector Addition: The Tip-to-Tail Method"
        lecture_lines = [
            "Adding vectors combines two different movements together.",
            "Place the second vector's tail at the first's tip.",
            "The new path starts at the very beginning.",
            "It ends at the tip of the second vector.",
            "This direct path is called the resultant vector."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors based on storyboard
        COLOR_A = "#00AAFF" # Blue
        COLOR_B = "#FF55FF" # Pink
        COLOR_RESULTANT = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        # Draw a blue vector A (#00AAFF) pointing upwards.
        self.play(self.lecture[0].animate.set_color(COLOR_A))
        
        # Vector A from E2 to C2 (Length 2)
        start_a = self.grid["E2"]
        end_a = self.grid["C2"]
        vector_a = Arrow(start_a, end_a, buff=0, color=COLOR_A)
        label_a = MathTex("\\vec{A}", color=COLOR_A)
        # Position label A to the left of the vector
        # ISSUE 28 FIXED: Moved from D1 to E1 to avoid crowding lecture notes
        self.place_at_grid(label_a, "E1", scale_factor=0.8)
        
        self.play(GrowArrow(vector_a), Write(label_a))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a pink vector B (#FF55FF) pointing right.
        self.play(self.lecture[1].animate.set_color(COLOR_B))
        
        # Vector B from E4 to E6 (Length 2)
        start_b = self.grid["E4"]
        end_b = self.grid["E6"]
        vector_b = Arrow(start_b, end_b, buff=0, color=COLOR_B)
        label_b = MathTex("\\vec{B}", color=COLOR_B)
        # Position label B above original vector B
        # ISSUE 26 FIXED: Moved to B4 as suggested by critic
        self.place_at_grid(label_b, "B4", scale_factor=0.8)
        
        self.play(GrowArrow(vector_b), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shift vector B so its tail touches vector A's tip.
        self.play(self.lecture[2].animate.set_color(COLOR_B))
        
        # Displacement vector: tail of B moves to tip of A
        shift_vector = end_a - start_b
        
        self.play(
            vector_b.animate.shift(shift_vector),
            label_b.animate.shift(shift_vector)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A white resultant vector (#FFFFFF) appears from start to end.
        self.play(self.lecture[3].animate.set_color(COLOR_RESULTANT))
        
        # Resultant from start of A to the new tip of B
        new_end_b = vector_b.get_end()
        vector_res = Arrow(start_a, new_end_b, buff=0, color=COLOR_RESULTANT)
        label_res = MathTex("\\vec{R}", color=COLOR_RESULTANT)
        # Position label R near the bottom right of the addition diagram
        # ISSUE 27 FIXED: Moved to E4 to avoid overlap with diagonal path
        self.place_at_grid(label_res, "E4", scale_factor=0.8)
        
        self.play(GrowArrow(vector_res), Write(label_res))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the formula 'A + B' (#FFFFFF) using a flash.
        self.play(self.lecture[4].animate.set_color(COLOR_RESULTANT))
        
        # Formula: R = A + B
        formula = MathTex("\\vec{R} = \\vec{A} + \\vec{B}", color=COLOR_RESULTANT)
        # Place in reserved row A area
        self.place_in_area(formula, "A3", "A5", scale_factor=0.9)
        
        self.play(Write(formula))
        self.play(Indicate(formula))
        self.wait(2)
