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
        
        # Colors
        color1 = "#FFFFFF"
        color2 = "#00FF00"
        color3 = "#00FF00" 

        # === Animation for Lecture Line 1 ===
        # Show a #FFFFFF arrow alongside its #FFFFFF [x, y] matrix.
        self.play(self.lecture[0].animate.set_color(color1))
        
        vec_arrow = Arrow(ORIGIN, RIGHT * 1.2, buff=0, color=color1)
        vec_matrix = MathTex(r"\begin{bmatrix} x \\ y \end{bmatrix}", color=color1)
        
        # FIX ISSUE 31: Move to B2 and B4 to avoid clutter and overlap
        self.place_at_grid(vec_arrow, "B2", scale_factor=1.0)
        self.place_at_grid(vec_matrix, "B4", scale_factor=1.2)
        
        self.play(
            GrowArrow(vec_arrow),
            Write(vec_matrix)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display a #00FF00 grid of arrows representing a force field, 
        # incorporating the physics icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/physics.svg].
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Group to hold the grid (3x3 for efficiency)
        force_field = VGroup()
        for i in range(3):
            for j in range(3):
                pos_x = (i - 1.0) * 0.5
                pos_y = (j - 1.0) * 0.5
                # Rotational field direction
                direction = np.array([-pos_y, pos_x, 0])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm * 0.3
                else:
                    direction = RIGHT * 0.3
                
                arrow = Arrow(ORIGIN, direction, buff=0, color=color2, stroke_width=2)
                arrow.move_to(np.array([pos_x, pos_y, 0]))
                force_field.add(arrow)
        
        # FIX ISSUE 32: Position force field in D2-F4 area to prevent obstruction
        self.place_in_area(force_field, "D2", "F4", scale_factor=0.8)
        
        # Load physics icon from specified path
        physics_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/physics.svg")
        physics_icon.set_color(color2)
        # FIX ISSUE 33: Move physics icon to E5 for better visual balance
        self.place_at_grid(physics_icon, "E5", scale_factor=1.0)

        self.play(
            FadeOut(vec_arrow),
            FadeOut(vec_matrix),
            LaggedStart(*[GrowArrow(a) for a in force_field], lag_ratio=0.05),
            FadeIn(physics_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scale the #00FF00 grid to show a linear transformation.
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Group content for transformation
        scene_content = VGroup(force_field, physics_icon)
        
        # Apply transformation (scaling and shear using apply_matrix to illustrate linear algebra concepts)
        # Shear along X-axis by 0.3: x' = x + 0.3y, y' = y
        shear_matrix = [[1, 0.3, 0], 
                        [0, 1, 0], 
                        [0, 0, 1]]
        
        self.play(
            scene_content.animate.scale(1.1).apply_matrix(shear_matrix),
            run_time=2
        )
        self.wait(3)
