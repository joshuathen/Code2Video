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
        self.setup_layout("Conclusion and Summary", [
            "A matrix represents a geometric instruction for space.",
            "Composing transformations equals matrix multiplication.",
            "Linear algebra is the study of such transformations."
        ])
        
        # Animation Elements
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False}).scale(0.6)
        square = Square(side_length=1, color=BLUE, fill_opacity=0.3)
        
        # FIX 38: Move plane to avoid overlap
        self.place_in_area(plane, 'D2', 'E5', scale_factor=1.0)
        self.place_at_grid(square, "D3", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # A matrix represents a geometric instruction for space.
        self.lecture[0].set_color("#FFFFFF")
        matrix_m = MathTex(r"M = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}").scale(0.8)
        # FIX 37: Move matrix formula to avoid clustering
        self.place_in_area(matrix_m, 'A3', 'B5', scale_factor=0.9)
        
        self.play(FadeIn(matrix_m))
        self.play(square.animate.apply_matrix([[2, 0], [0, 1]]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Composing transformations equals matrix multiplication.
        self.lecture[1].set_color("#00FFFF")
        comp_text = MathTex(r"M_2 \cdot M_1 = M_{total}").scale(0.8)
        self.place_at_grid(comp_text, "C4")
        
        self.play(FadeIn(comp_text))
        # Shear transformation as composition
        self.play(square.animate.apply_matrix([[1, 1], [0, 1]]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Linear algebra is the study of such transformations.
        self.lecture[2].set_color("#FF00FF")
        final_text = Text("Linear Transformations", font_size=24, color="#FF00FF")
        # FIX 39: Move label up
        self.place_at_grid(final_text, 'F2', scale_factor=0.9)
        
        self.play(Write(final_text))
        self.play(FadeOut(square), FadeOut(plane), FadeOut(matrix_m), FadeOut(comp_text))
        self.wait(2)
