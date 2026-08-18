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
        self.setup_layout("Non-Traditional Examples: Beyond Arrows", [
            "Unusual sets can form vector spaces.",
            "Polynomials act like coordinate-based vectors.",
            "Degree-two polynomials behave identically to 3D space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display function set labeled #FFFFFF
        set_label = Text("Set P: {a + bx + cx^2}", font_size=24, color=WHITE)
        self.place_at_grid(set_label, 'C2', scale_factor=0.9)
        self.play(Write(set_label))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Animate functions as arrows in #FF4500
        vec = Vector([1, 1, 0.5], color="#FF4500")
        vec_label = MathTex(r"\\begin{bmatrix} a \\\\ b \\\\ c \\end{bmatrix}", color="#FF4500", font_size=24)
        self.place_at_grid(vec, 'D2', scale_factor=0.8)
        self.place_at_grid(vec_label, 'D3', scale_factor=0.8)
        
        self.play(Create(vec), Write(vec_label))
        self.lecture[1].set_color("#FF4500")
        
        # === Animation for Lecture Line 3 ===
        # Show matrix as grid in #40E0D0
        matrix_grid = Matrix([[r"a"], [r"b"], [r"c"]], color="#40E0D0")
        self.place_in_area(matrix_grid, 'E3', 'F4', scale_factor=0.8)
        
        self.play(Create(matrix_grid))
        self.lecture[2].set_color("#40E0D0")
        self.wait(2)
