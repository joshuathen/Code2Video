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
        title_text = "The Change of Basis Matrix (The Translator)"
        lecture_lines = [
            "Matrix P acts as a translator between these grids.",
            "We build P using the new basis vectors.",
            "Each new vector becomes a column in Matrix P.",
            "These columns are written in the standard basis language.",
            "This matrix connects the robot's world to ours."
        ]
        self.setup_layout(title_text, lecture_lines)

        def create_vector_mobject(coords, color):
            # Creates a column vector visual [x, y] using Text
            v_elements = VGroup(
                Text(str(coords[0]), font_size=24),
                Text(str(coords[1]), font_size=24)
            ).arrange(DOWN, buff=0.3)
            bracket_l = Text("[", font_size=40).stretch_to_fit_height(v_elements.height + 0.2)
            bracket_r = Text("]", font_size=40).stretch_to_fit_height(v_elements.height + 0.2)
            bracket_l.next_to(v_elements, LEFT, buff=0.1)
            bracket_r.next_to(v_elements, RIGHT, buff=0.1)
            return VGroup(bracket_l, v_elements, bracket_r).set_color(color)

        # Initial state of lecture lines (all dimmed except first)
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Matrix P acts as a translator between these grids.
        change_basis_text = Text("Change of Basis Matrix P:", font_size=24)
        # Fix for Issue 28: self.place_in_area(change_basis_text, 'A2', 'B5', scale_factor=0.8)
        self.place_in_area(change_basis_text, 'A2', 'B5', scale_factor=0.8)
        self.play(Write(change_basis_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We build P using the new basis vectors.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        b1 = create_vector_mobject([2, 1], "#00FFFF")
        b2 = create_vector_mobject([-1, 1], "#FFFF00")
        
        b1_label = Text("b1", font_size=20, color="#00FFFF").next_to(b1, UP, buff=0.1)
        b2_label = Text("b2", font_size=20, color="#FFFF00").next_to(b2, UP, buff=0.1)
        
        vecs = VGroup(VGroup(b1, b1_label), VGroup(b2, b2_label)).arrange(RIGHT, buff=1.0)
        self.place_in_area(vecs, 'C2', 'D5', scale_factor=1.0)
        
        self.play(FadeIn(vecs))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Each new vector becomes a column in Matrix P.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Construct the matrix P from the vectors
        # Values arranged in rows
        r1_c1 = Text("2", font_size=24)
        r1_c2 = Text("-1", font_size=24)
        r2_c1 = Text("1", font_size=24)
        r2_c2 = Text("1", font_size=24)
        
        row1 = VGroup(r1_c1, r1_c2).arrange(RIGHT, buff=0.8)
        row2 = VGroup(r2_c1, r2_c2).arrange(DOWN, buff=0.5) # Dummy setup for spacing
        
        p_vals = VGroup(
            VGroup(r1_c1, r1_c2).arrange(RIGHT, buff=0.8),
            VGroup(r2_c1, r2_c2).arrange(RIGHT, buff=0.8)
        ).arrange(DOWN, buff=0.5)
        
        p_bracket_l = Text("[", font_size=60).stretch_to_fit_height(p_vals.height + 0.3)
        p_bracket_r = Text("]", font_size=60).stretch_to_fit_height(p_vals.height + 0.3)
        p_matrix = VGroup(p_bracket_l, p_vals, p_bracket_r)
        p_bracket_l.next_to(p_vals, LEFT, buff=0.2)
        p_bracket_r.next_to(p_vals, RIGHT, buff=0.2)
        
        # Fix for Issue 29: self.place_in_area(p_matrix, 'C2', 'E5', scale_factor=1.2)
        self.place_in_area(p_matrix, 'C2', 'E5', scale_factor=1.2)
        
        # Identify columns for coloring
        col1 = VGroup(p_vals[0][0], p_vals[1][0]).set_color("#00FFFF")
        col2 = VGroup(p_vals[0][1], p_vals[1][1]).set_color("#FFFF00")
        
        self.play(
            FadeOut(vecs),
            FadeIn(p_bracket_l), FadeIn(p_bracket_r),
            Write(col1), Write(col2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # These columns are written in the standard basis language.
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(WHITE)
        )
        
        highlight_col1 = SurroundingRectangle(col1, color="#00FFFF", buff=0.1)
        highlight_col2 = SurroundingRectangle(col2, color="#FFFF00", buff=0.1)
        
        col_label = Text("New Basis in Old Terms", font_size=18, color=WHITE)
        # Use place_at_grid to position label near the matrix
        self.place_at_grid(col_label, 'F3', scale_factor=1.0)
        
        self.play(Create(highlight_col1))
        self.play(ReplacementTransform(highlight_col1, highlight_col2))
        self.play(Write(col_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This matrix connects the robot's world to ours.
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        full_box = SurroundingRectangle(p_matrix, color=WHITE, buff=0.2)
        transition_label = Text("Transition Matrix", font_size=20, color=WHITE)
        self.place_at_grid(transition_label, 'F4', scale_factor=1.0)
        
        self.play(FadeOut(highlight_col2), FadeOut(col_label))
        self.play(Create(full_box))
        self.play(Write(transition_label))
        self.wait(2)
