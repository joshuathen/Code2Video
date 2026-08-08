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
        self.setup_layout("Non-Geometric Examples: Polynomials", [
            "Polynomials can act like vectors.", 
            "Adding polynomials mirrors vector addition.", 
            "Scaling preserves the polynomial's degree."
        ])
        
        # Polynomial P(x) = ax^2 + bx + c
        # Update: Using place_in_area for better positioning as requested in issue 40/25
        poly = MathTex("P(x) = a", "x^2", "+", "b", "x", "+", "c")
        self.place_in_area(poly, 'A3', 'B5', scale_factor=1.0)
        
        # Vector version
        # Update: Using place_at_grid('D3', 0.9) for better visual grouping as requested in issue 41/27/26
        vec = MathTex("\\vec{v} = \\begin{bmatrix} a \\\\ b \\\\ c \\end{bmatrix}")
        self.place_at_grid(vec, 'D3', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(poly))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight coefficients a, b, c
        self.play(
            Indicate(poly[0:1], color="#FFFF00"),
            Indicate(poly[3:4], color="#FFFF00"),
            Indicate(poly[6:7], color="#FFFF00"),
            run_time=2
        )
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show them behaving like a vector
        self.play(Write(vec))
        self.play(
            poly[0:1].animate.set_color("#00FF00"),
            poly[3:4].animate.set_color("#00FF00"),
            poly[6:7].animate.set_color("#00FF00"),
            vec.animate.set_color("#00FF00")
        )
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
