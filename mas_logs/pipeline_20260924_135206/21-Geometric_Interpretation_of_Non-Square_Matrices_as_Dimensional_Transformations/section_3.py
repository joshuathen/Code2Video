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
        lecture_lines = [
            "Wide matrices map high dimensions down to low.",
            "This projection compresses data, sacrificing depth.",
            "Like a 3D object captured on 2D film."
        ]
        self.setup_layout("Mapping Down: The Concept of Projection", lecture_lines)
        
        # Create objects
        # Using SVG asset for cube
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#44CCFF")
        matrix = MathTex(r"A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}", color=WHITE)
        plane = Square(side_length=2, fill_opacity=0.3, color="#FFCC44")
        
        # Positions adjusted per issue fixes
        self.place_at_grid(cube, "B3", scale_factor=0.6)
        self.place_in_area(matrix, "D4", "F6", scale_factor=0.9)
        self.place_at_grid(plane, "A4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#44CCFF"))
        self.play(Create(cube), Write(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC44"))
        self.play(FadeIn(plane))
        # Project cube shadow onto plane
        shadow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#FFCC44", fill_opacity=0.5)
        self.place_at_grid(shadow, "A4", scale_factor=0.4)
        
        self.play(
            cube.animate.scale(0.5).move_to(self.grid["A4"]),
            FadeIn(shadow)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
