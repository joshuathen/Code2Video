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
        self.setup_layout("The Classic Proof: Euler's Characteristic", [
            "Euler's formula relates vertices, edges, and faces.", 
            "V minus E plus F always equals two.", 
            "This formula holds for any convex polyhedron."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display vertex and edge
        dot = Dot(color="#00FFFF")
        line = Line(start=ORIGIN, end=RIGHT, color="#00FFFF")
        self.place_at_grid(dot, "B4", scale_factor=0.6)
        self.place_at_grid(line, "B3", scale_factor=0.6)
        self.play(Create(dot), Create(line))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        # Construct cube frame using SVG asset
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_at_grid(cube, "E5", scale_factor=0.5)
        self.play(Create(cube))
        
        formula = MathTex(r"V - E + F = 2", font_size=48, color="#FF69B4")
        self.place_in_area(formula, "A4", "B6", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[1].set_color("#FF69B4")

        # === Animation for Lecture Line 3 ===
        # Highlight components and the constant 2
        # Emphasize the constant value 2 in bright green (#00FF00) for the cube
        highlighted_cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_at_grid(highlighted_cube, "E5", scale_factor=0.5)
        highlighted_cube.set_color("#00FF00")
        
        self.play(
            formula.animate.set_color("#00FF00"),
            ReplacementTransform(cube, highlighted_cube)
        )
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
