from manim import *
import numpy as np

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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Real-World Application", [
            "Vectors are both geometric arrows and numerical lists.",
            "They power physics engines, graphics, and data science.",
            "Mastering vectors unlocks the heart of linear algebra."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display a vector field with multiple #FFFFFF arrows pointing in different directions.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        arrows = VGroup()
        for i in range(4):
            for j in range(4):
                # Deterministic rotation for visual variety
                angle = (i * 0.7 + j * 1.3) % (2 * PI)
                arrow = Arrow(ORIGIN, 0.4 * RIGHT, buff=0, color="#FFFFFF", stroke_width=3)
                arrow.rotate(angle)
                arrow.move_to(np.array([i*0.8, j*0.8, 0]))
                arrows.add(arrow)
        
        self.place_in_area(arrows, "B2", "E5", scale_factor=0.9)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight one arrow and transform it into a [x, y] column matrix.
        # Matching color for Line 2 and its matrix transformation.
        line2_color = "#FFFF00" # Yellow
        self.play(self.lecture[1].animate.set_color(line2_color))
        
        # Pick a central arrow to transform
        target_arrow = arrows[6] 
        
        # Pre-create matrix to avoid complex mobjects in updaters (though not using updaters here)
        matrix = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix}", color=line2_color)
        self.place_at_grid(matrix, "C4", scale_factor=1.5)

        self.play(
            target_arrow.animate.set_color(line2_color).scale(1.2),
            *[FadeOut(a) for a in arrows if a != target_arrow]
        )
        
        self.play(Transform(target_arrow, matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a simple #00FF00 wireframe mesh to represent a 3D model.
        line3_color = "#00FF00" # Green
        self.play(self.lecture[2].animate.set_color(line3_color))
        
        # Create a simple mesh-like structure
        mesh = VGroup()
        # Horizontal lines
        for i in range(6):
            l = Line([-1.5, i*0.5 - 1.25, 0], [1.5, i*0.5 - 1.25, 0], color=line3_color, stroke_width=1.5)
            mesh.add(l)
        # Vertical lines
        for i in range(6):
            l = Line([i*0.5 - 1.25, -1.5, 0], [i*0.5 - 1.25, 1.5, 0], color=line3_color, stroke_width=1.5)
            mesh.add(l)
        # Diagonal lines to give a 3D wireframe mesh aesthetic
        for i in range(5):
            for j in range(5):
                l = Line([i*0.5 - 1.25, j*0.5 - 1.25, 0], [(i+1)*0.5 - 1.25, (j+1)*0.5 - 1.25, 0], 
                         color=line3_color, stroke_width=0.8, stroke_opacity=0.6)
                mesh.add(l)

        self.place_in_area(mesh, "B2", "E5", scale_factor=1.0)
        
        self.play(
            FadeOut(target_arrow),
            Create(mesh, run_time=2)
        )
        self.wait(3)
