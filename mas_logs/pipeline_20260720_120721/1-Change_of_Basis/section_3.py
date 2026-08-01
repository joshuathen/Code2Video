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
        lecture_lines = [\
            "The change-of-basis matrix converts coordinates between bases.",\
            "It\\'s constructed using the new basis vectors as columns.",\
            "These vectors are expressed in terms of the original basis.",\
            "Multiplying a vector\\'s old coordinates by this matrix yields new coordinates.",\
            "This matrix is crucial for transformations involving different bases.",\
        ]
        self.setup_layout("The Transition Matrix: The Bridge", lecture_lines)
        
        # Define colors for lecture lines
        color1 = "#FFD700"  # Gold
        color2 = "#87CEEB"  # SkyBlue
        color3 = "#ADFF2F"  # GreenYellow
        color4 = "#FF6347"  # Tomato
        color5 = "#DA70D6"  # Orchid

        # === Animation for Lecture Line 1 ===
        # Appear: The transition matrix [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg] connects two bases.
        self.play(self.lecture[0].animate.set_color(color1))
        
        transition_matrix_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        self.place_at_grid(transition_matrix_icon, 'C3', scale_factor=1.0)
        
        # Representing "connects two bases" conceptually with text and arrows
        basis_e_text = Text("Standard Basis E", font_size=20, color=WHITE)
        basis_b_text = Text("New Basis B", font_size=20, color=WHITE)

        self.place_at_grid(basis_e_text, 'B1', scale_factor=0.8)
        self.place_at_grid(basis_b_text, 'D5', scale_factor=0.8)

        arrow_e_to_matrix = Arrow(start=basis_e_text.get_right(), end=transition_matrix_icon.get_left(), buff=0.1)
        arrow_matrix_to_b = Arrow(start=transition_matrix_icon.get_right(), end=basis_b_text.get_left(), buff=0.1)

        self.play(FadeIn(transition_matrix_icon), FadeIn(basis_e_text), FadeIn(basis_b_text), Create(arrow_e_to_matrix), Create(arrow_matrix_to_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight: It [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg] allows conversion from one basis to another.
        self.play(
            self.lecture[1].animate.set_color(color2),
            Flash(transition_matrix_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scale up: It [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg]s formed by the new basis vectors in old coordinates.
        # Fade in: Show the original basis vectors.
        # Fade in: Show the new basis vectors and the transition matrix [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg].
        self.play(
            self.lecture[2].animate.set_color(color3),
            transition_matrix_icon.animate.scale(1.2)
        )

        # Original basis vectors (e.g., standard basis)
        e1 = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=RED)
        e2 = Arrow(start=ORIGIN, end=UP, buff=0, color=BLUE)
        # Fix: Replaced MathTex with Text to resolve FileNotFoundError: 'latex' if LaTeX is not installed.
        e1_label = Text("e_1", color=RED).next_to(e1, RIGHT, buff=0.1)
        # Fix: Changed MathTex to Text to resolve FileNotFoundError: 'latex'
        e2_label = Text("e_2", color=BLUE).next_to(e2, UP, buff=0.1)
        original_basis = VGroup(e1, e2, e1_label, e2_label)
        self.place_at_grid(original_basis, 'E2', scale_factor=1.0)

        self.play(FadeIn(original_basis))

        # New basis vectors (example)
        v1 = Arrow(start=ORIGIN, end=[2, 1, 0], buff=0, color=GREEN)
        v2 = Arrow(start=ORIGIN, end=[1, -1, 0], buff=0, color=PURPLE)
        # Fix: Replaced MathTex with Text to resolve FileNotFoundError: 'latex' if LaTeX is not installed.
        v1_label = Text("v_1", color=GREEN).next_to(v1.get_end(), RIGHT, buff=0.1)
        v2_label = Text("v_2", color=PURPLE).next_to(v2.get_end(), UP, buff=0.1)
        new_basis = VGroup(v1, v2, v1_label, v2_label)
        self.place_at_grid(new_basis, 'E4', scale_factor=1.0)
        
        self.play(FadeIn(new_basis), FadeIn(transition_matrix_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Multiplying a vector's old coordinates by this matrix yields new coordinates.
        self.play(self.lecture[3].animate.set_color(color4))

        # Introduce a vector in the standard basis
        w_vector = Arrow(start=ORIGIN, end=[3, 2, 0], buff=0, color=YELLOW)
        # Fix: Changed MathTex to Text to resolve FileNotFoundError: 'latex'
        w_label = Text("w", color=YELLOW).next_to(w_vector.get_end(), UP, buff=0.1)
        self.place_at_grid(VGroup(w_vector, w_label), 'A3', scale_factor=0.8)
        self.play(FadeIn(w_vector), FadeIn(w_label))
        self.wait(0.5)

        # Show the transformation equation
        # FIX: Changed MathTex to Text and simplified arguments to avoid FileNotFoundError: 'latex'
        transformation_eq = Text("w_B = P^{-1} w_E", color=WHITE) 
        # The following lines rely on MathTex's internal LaTeX parsing for substrings_to_isolate
        # and set_color_by_tex. Since we switched to Text, these are removed or handled differently.
        # For a Text object, setting the color in the constructor applies to the entire text.
        # If fine-grained coloring is needed without LaTeX, consider using Text with span tags or creating multiple Text mobjects.
        # For this fix, we are prioritizing fixing the FileNotFoundError by removing LaTeX dependency.
        # transformation_eq.set_color_by_tex("w_B", YELLOW) 
        # transformation_eq.set_color_by_tex("P^{-1}", WHITE)
        # transformation_eq.set_color_by_tex("w_E", YELLOW)
        self.place_at_grid(transformation_eq, 'B3', scale_factor=0.9)
        self.play(Write(transformation_eq))
        self.wait(1)

        # Emphasize the transformation visually by moving the vector
        # Conceptually, w_B would be the same vector but with different coordinates in basis B
        # For visualization, we can show w_vector morphing or moving to represent its new "view" in basis B.
        # However, to keep it simple, let's just show new coordinates.
        # Alternatively, for simplicity, just show the equation and then highlight the matrix role.
        
        # A simple animation of the matrix being applied to the vector
        self.play(
            w_vector.animate.shift(0.5 * RIGHT + 0.5 * DOWN).set_color(ORANGE),
            w_label.animate.next_to(w_vector.get_end(), DOWN, buff=0.1).set_color(ORANGE),
            Flash(transition_matrix_icon, color=WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This matrix is crucial for transformations involving different bases.
        self.play(self.lecture[4].animate.set_color(color5))
        
        # Re-emphasize the transition matrix and its role between bases
        self.play(
            Flash(transition_matrix_icon, color=WHITE, line_stroke_width=3),
            basis_e_text.animate.move_to(self.grid['B1']), # Fix: Removed redundant FadeIn
            basis_b_text.animate.move_to(self.grid['D5']), # Fix: Removed redundant FadeIn
            Create(arrow_e_to_matrix),
            Create(arrow_matrix_to_b),
            FadeOut(original_basis),
            FadeOut(new_basis),
            FadeOut(w_vector),
            FadeOut(w_label),
            FadeOut(transformation_eq)
        )
        self.wait(2)
