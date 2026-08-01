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

class Section1Scene(TeachingScene):
    def construct(self):
        title = "What is a Basis?"
        lecture_lines = [
            "- What is a basis?",
            "- Linearly independent vectors spanning a space.",
            "- Unique representation of any vector.",
            "- Standard basis e1=[1,0], e2=[0,1].",
            "- Coordinates are coefficients of basis vectors."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors for lecture lines
        color_line_1 = "#FFD700" # Gold
        color_line_2 = "#87CEEB" # Sky Blue
        color_line_3 = "#98FB98" # Pale Green
        color_line_4 = "#FF6347" # Tomato
        color_line_5 = "#DDA0DD" # Plum

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_line_1))
        
        basis_def = Text("A basis is a set of linearly independent vectors that span the entire space.", font_size=28, color=color_line_1)
        self.place_in_area(basis_def, 'A1', 'C6', scale_factor=0.7)
        self.play(FadeIn(basis_def))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(Transform(self.lecture[0], self.lecture[0].copy().set_color(WHITE)),
                  self.lecture[1].animate.set_color(color_line_2))

        # Highlighting "linearly independent" part
        linearly_independent = Text("linearly independent", font_size=28, color=color_line_2)
        self.place_at_grid(linearly_independent, 'D3', scale_factor=0.8)
        self.play(basis_def.animate.become(Text("A basis is a set of linearly independent vectors that span the entire space.", font_size=28, color=color_line_1)),
                  FadeIn(linearly_independent))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(Transform(self.lecture[1], self.lecture[1].copy().set_color(WHITE)),
                  self.lecture[2].animate.set_color(color_line_3))
        
        # Fix: Replaced MathTex with MarkupText to resolve FileNotFoundError: 'latex'.
        # This bypasses the need for a LaTeX installation by using Manim's internal text rendering with markup for formatting.
        vector_representation = MarkupText(r"<b>v</b> = c<sub>1</sub><b>b</b><sub>1</sub> + c<sub>2</sub><b>b</b><sub>2</sub> + ... + c<sub>n</sub><b>b</b><sub>n</sub>", font_size=32, color=color_line_3)
        self.place_in_area(vector_representation, 'E1', 'F6', scale_factor=0.8)

        self.play(FadeOut(basis_def),
                  FadeOut(linearly_independent),
                  FadeIn(vector_representation))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(Transform(self.lecture[2], self.lecture[2].copy().set_color(WHITE)),
                  self.lecture[3].animate.set_color(color_line_4))

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": GRAY}
        )
        self.place_in_area(axes, 'A1', 'D6', scale_factor=0.8)
        
        e1_arrow = Arrow(self.grid['C3'], self.grid['C4'], buff=0, max_stroke_width_to_length_ratio=8, color=color_line_4)
        e1_label = MarkupText(r"<b>e</b><sub>1</sub> = [1,0]", color=color_line_4).next_to(e1_arrow.get_end(), RIGHT)
        self.place_at_grid(e1_arrow, 'C3')
        self.place_at_grid(e1_label, 'C5', scale_factor=0.8)

        e2_arrow = Arrow(self.grid['C3'], self.grid['B3'], buff=0, max_stroke_width_to_length_ratio=8, color=color_line_4)
        e2_label = MarkupText(r"<b>e</b><sub>2</sub> = [0,1]", color=color_line_4).next_to(e2_arrow.get_end(), UP)
        self.place_at_grid(e2_arrow, 'B3')
        self.place_at_grid(e2_label, 'A3', scale_factor=0.8)

        self.play(FadeOut(vector_representation),
                  Create(axes),
                  GrowArrow(e1_arrow), FadeIn(e1_label))
        self.wait(1)
        self.play(GrowArrow(e2_arrow), FadeIn(e2_label))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(Transform(self.lecture[3], self.lecture[3].copy().set_color(WHITE)),
                  self.lecture[4].animate.set_color(color_line_5))
        
        # Using MarkupText for the equation to avoid latex dependency
        coords_equation = MarkupText(r"<b>v</b> = x<b>e</b><sub>1</sub> + y<b>e</b><sub>2</sub>", font_size=32, color=color_line_5)
        self.place_in_area(coords_equation, 'E1', 'F6', scale_factor=0.8)

        # For TransformMatchingTex with MarkupText, it's generally better to define the target MarkupText directly.
        # coords_equation[2] and coords_equation[5] would refer to VGroup sub-mobjects which might not match precisely
        # after changing from MathTex.
        # The following lines for x_coord and y_coord would need to be re-evaluated if a precise character match is required
        # For simplicity and to fit the single-line scope, the `TransformMatchingTex` logic might need adjustment.
        # As `TransformMatchingTex` is for Tex/MathTex, for MarkupText, a simple Transform or custom animation might be needed.
        # For the current error, ensuring `coords_equation` is correctly created without LaTeX is the priority.
        
        # The subsequent TransformMatchingTex needs Tex or MathTex usually for character-level matching.
        # Changing this part too would extend beyond the single-line fix for the 'latex' error.
        # If `TransformMatchingTex` fails with MarkupText, it would be a new error to address.
        # For the current error, ensuring `coords_equation` is correctly created without LaTeX is the priority.
        
        # Assuming TransformMatchingTex might not work ideally with MarkupText, but for the immediate fix,
        # we'll create the target mobject for the transformation.
        target_coords_equation = MarkupText(r"<b>v</b> = <b>x</b><b>e</b><sub>1</sub> + <b>y</b><b>e</b><sub>2</sub>", font_size=32, color=color_line_5)
        self.place_in_area(target_coords_equation, 'E1', 'F6', scale_factor=0.8)

        self.play(FadeIn(coords_equation))
        self.wait(1)
        # TransformMatchingTex works best with Tex/MathTex. If it causes issues with MarkupText,
        # a simpler `Transform` or manual replacement might be necessary.
        # The `AssertionError` occurred because `TransformMatchingTex` was used with `MarkupText`
        # which lacks the `tex_string` attribute. Replacing it with `Transform` is the correct fix.
        self.play(Transform(coords_equation, target_coords_equation)) 
        self.wait(2)
        
        # Fade out everything
        self.play(FadeOut(axes), FadeOut(e1_arrow), FadeOut(e1_label), FadeOut(e2_arrow), FadeOut(e2_label), FadeOut(coords_equation))
        self.wait(1)
