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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines from storyboard
        self.setup_layout("Prerequisite Review: Basis and Coordinates", [
            "A basis consists of linearly independent spanning vectors.",
            "Any vector is a recipe of these basis elements.",
            "Standard basis vectors i and j form our grid."
        ])
        
        # Colors
        color_i = "#FF0000"
        color_j = "#00FF00"
        color_result = "#FFFF00"
        color_grid = "#444444"

        # 1. Create Basis Vectors and grid
        # Local coordinate system: 1 unit = 1 grid cell width
        origin_loc = ORIGIN
        i_hat = Arrow(origin_loc, RIGHT, buff=0, color=color_i, stroke_width=4)
        j_hat = Arrow(origin_loc, UP, buff=0, color=color_j, stroke_width=4)
        
        # Background grid for the diagram
        grid_lines = VGroup()
        for x in range(4):
            grid_lines.add(Line(x*RIGHT + 0*UP, x*RIGHT + 2*UP, color=color_grid, stroke_width=1, stroke_opacity=0.5))
        for y in range(3):
            grid_lines.add(Line(0*RIGHT + y*UP, 3*RIGHT + y*UP, color=color_grid, stroke_width=1, stroke_opacity=0.5))
            
        # Recipe vectors (3i + 2j)
        r_i1 = Arrow(0*RIGHT + 0*UP, 1*RIGHT + 0*UP, buff=0, color=color_i, stroke_width=2)
        r_i2 = Arrow(1*RIGHT + 0*UP, 2*RIGHT + 0*UP, buff=0, color=color_i, stroke_width=2)
        r_i3 = Arrow(2*RIGHT + 0*UP, 3*RIGHT + 0*UP, buff=0, color=color_i, stroke_width=2)
        r_j1 = Arrow(3*RIGHT + 0*UP, 3*RIGHT + 1*UP, buff=0, color=color_j, stroke_width=2)
        r_j2 = Arrow(3*RIGHT + 1*UP, 3*RIGHT + 2*UP, buff=0, color=color_j, stroke_width=2)
        
        # Result vector
        vec_32 = Arrow(origin_loc, 3*RIGHT + 2*UP, buff=0, color=color_result, stroke_width=6)
        
        # Group everything for Issue 25
        # We include all vectors in the group so they scale and move together
        vector_group = VGroup(grid_lines, i_hat, j_hat, r_i1, r_i2, r_i3, r_j1, r_j2, vec_32)
        
        # Labels
        i_label = MathTex(r"\hat{i}", color=color_i, font_size=36)
        j_label = MathTex(r"\hat{j}", color=color_j, font_size=36)
        coord_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=WHITE, font_size=42)
        
        # Anchoring to the grid system (Issues 25, 26, 27)
        self.place_in_area(vector_group, 'C3', 'F6', scale_factor=0.9)
        self.place_at_grid(i_label, 'F4', scale_factor=0.6)
        # Based on origin alignment (E3), j_hat tip is near D3.
        self.place_at_grid(j_label, 'D2', scale_factor=0.6) 
        self.place_at_grid(coord_label, 'B6', scale_factor=0.8)

        # Initial state: hide everything in vector_group
        for obj in vector_group:
            obj.set_opacity(0)
        i_label.set_opacity(0)
        j_label.set_opacity(0)
        coord_label.set_opacity(0)

        # Add objects to scene (so they can be animated)
        self.add(vector_group, i_label, j_label, coord_label)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line and show basis vectors
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(
            i_hat.animate.set_opacity(1),
            j_hat.animate.set_opacity(1),
            Write(i_label.set_opacity(1)),
            Write(j_label.set_opacity(1)),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line and show vector addition steps
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(grid_lines.animate.set_opacity(1))
        
        # Animate steps head-to-tail
        self.play(Create(r_i1.set_opacity(1)), run_time=0.6)
        self.play(Create(r_i2.set_opacity(1)), run_time=0.6)
        self.play(Create(r_i3.set_opacity(1)), run_time=0.6)
        self.play(Create(r_j1.set_opacity(1)), run_time=0.6)
        self.play(Create(r_j2.set_opacity(1)), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line and show final vector with label
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(Create(vec_32.set_opacity(1)))
        self.play(Write(coord_label.set_opacity(1)))
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
