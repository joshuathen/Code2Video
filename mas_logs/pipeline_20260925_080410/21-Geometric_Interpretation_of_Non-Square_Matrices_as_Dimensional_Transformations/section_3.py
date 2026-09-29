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
        self.setup_layout("Dimensional Compression (Wide Matrices)", [
            "Wide matrices have more columns than rows.",
            "They compress higher dimensions into lower ones.",
            "This process is called linear projection.",
            "Like a camera flattening 3D to 2D.",
            "Information is lost during this compression."
        ])
        
        # Create visual elements
        matrix = MathTex(r"\\begin{bmatrix} a & b & c \\\\ d & e & f \\end{bmatrix}", color=BLUE)
        space_n = Text("R^3 (Input)", font_size=24, color=YELLOW)
        space_m = Text("R^2 (Output)", font_size=24, color=GREEN)
        vector_v = Arrow(start=ORIGIN, end=RIGHT*0.5+UP*0.5, color=WHITE)
        projection_line = Line(start=np.array([0,0,0]), end=np.array([0,0,0.5]), color=RED)
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(matrix, 'B2', scale_factor=1.2)
        self.play(FadeIn(matrix))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(space_n, 'B4', scale_factor=0.6)
        self.place_at_grid(space_m, 'E2', scale_factor=0.6)
        self.play(FadeIn(space_n), FadeIn(space_m))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.place_at_grid(vector_v, 'C3', scale_factor=0.8)
        self.play(Create(vector_v))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        self.place_at_grid(camera_icon, 'C5', scale_factor=0.5)
        self.place_in_area(projection_line, 'C3', 'D4', scale_factor=1.0)
        self.play(FadeIn(camera_icon), Create(projection_line))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(ORANGE))
        self.play(FadeOut(projection_line), FadeOut(vector_v), FadeOut(camera_icon))
        self.wait(1)
