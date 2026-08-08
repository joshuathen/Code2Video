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
        # Data from storyboard
        title = "Summary & Global View"
        lines = [
            "Change of basis maps grids to different perspectives.",
            "It is the foundation for diagonalization and compression.",
            "One vector, many views: that is change of basis."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Sequence: Basis -> Grid -> Matrix -> Translation (white #FFFFFF)
        # Issue 38 Fix: self.place_in_area(sequence, 'B2', 'B5', scale_factor=0.8)
        
        basis_txt = Text("Basis", font_size=24, color=WHITE)
        grid_txt = Text("Grid", font_size=24, color=WHITE)
        matrix_txt = Text("Matrix", font_size=24, color=WHITE)
        trans_txt = Text("Translation", font_size=24, color=WHITE)
        
        arrow1 = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1)
        arrow2 = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1)
        arrow3 = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1)
        
        sequence = VGroup(
            basis_txt, arrow1, grid_txt, arrow2, matrix_txt, arrow3, trans_txt
        ).arrange(RIGHT, buff=0.2)
        
        # Grid positioning: Avoid Row A/F, maintain gap from lecture text (Column 1)
        self.place_in_area(sequence, 'B2', 'B5', scale_factor=0.8)
        
        # Line 1 is active (White matches sequence)
        self.play(Write(sequence))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display 'Diagonalization' and 'Compression' as key applications in green (#00FF00).
        # Issue 37 Fix: self.place_in_area(apps, 'E2', 'E5', scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        diag_txt = Text("Diagonalization", font_size=24, color="#00FF00")
        comp_txt = Text("Compression", font_size=24, color="#00FF00")
        apps = VGroup(diag_txt, comp_txt).arrange(RIGHT, buff=0.8)
        
        self.place_in_area(apps, 'E2', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(apps))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Morph the tilted grid back to standard while the vector arrow remains fixed.
        # Issue 36 Fix: self.place_in_area(v_grid, 'C2', 'D6', scale_factor=0.65)
        
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Prepare Grid and Vector
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_D, "stroke_width": 1, "stroke_opacity": 0.5}
        )
        # Apply a tilted basis transformation initially
        matrix = np.array([[1.5, 0.5], [0.5, 1.5]])
        plane.apply_matrix(matrix)
        
        # Vector remains fixed in space relative to the scene
        vector = Vector([1, 1], color=YELLOW)
        
        # Group them to place centrally in the designated area
        v_grid = VGroup(plane, vector)
        self.place_in_area(v_grid, 'C2', 'D6', scale_factor=0.65)
        
        self.play(Create(plane), GrowArrow(vector))
        self.wait(1)
        
        # Morph back to standard identity basis
        self.play(
            plane.animate.apply_matrix(np.linalg.inv(matrix)),
            run_time=3
        )
        self.wait(2)
