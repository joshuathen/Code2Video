from manim import *
import os

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
        self.setup_layout("Summary & Synthesis", [
            "3D transformations remap the entire 3D space grid.",
            "The matrix is a recipe for basis movement.",
            "Multiplication tracks complex character motion smoothly."
        ])
        
        # Asset definition
        char_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png"
        char_icon = ImageMobject(char_path) if os.path.exists(char_path) else Dot(color=BLUE)

        # === Animation for Lecture Line 1 ===
        # 3D grid remapping
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2]).rotate(PI/4, axis=OUT).rotate(PI/6, axis=RIGHT)
        self.place_in_area(grid, 'B2', 'D6', scale_factor=0.4)
        
        # Add character to the grid
        char_icon_1 = char_icon.copy()
        self.place_at_grid(char_icon_1, 'C5', scale_factor=0.3)
        
        self.play(Create(grid), FadeIn(char_icon_1), self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matrix as a recipe
        matrix = MathTex(r"\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}")
        self.place_in_area(matrix, 'D4', 'F6', scale_factor=0.6)
        
        # Determinant formula
        det_formula = MathTex(r"\det(M) = \text{volume scaling factor}")
        self.place_at_grid(det_formula, 'E2', scale_factor=0.5)
        
        self.play(Write(matrix), Write(det_formula), self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Character motion representation
        char_icon_2 = char_icon.copy()
        self.place_at_grid(char_icon_2, 'F2', scale_factor=0.4)
        
        self.play(char_icon_2.animate.shift(RIGHT * 1.5 + UP * 0.5), self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
