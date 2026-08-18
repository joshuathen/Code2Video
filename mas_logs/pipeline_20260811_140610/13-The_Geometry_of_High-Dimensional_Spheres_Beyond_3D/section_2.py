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
        lecture_lines = [
            "Now we generalize to N-dimensions, the hypersphere.",
            "The equation follows the same sum-of-squares pattern.",
            "We represent an N-dimensional unit sphere algebraically."
        ]
        self.setup_layout("Generalizing to N-Dimensions", lecture_lines)
        
        # Elements
        n_cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=WHITE)
        n_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        formula = MathTex(r"\sum_{i=1}^{n} x_i^2 = r^2", color="#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        # Now we generalize to N-dimensions, the hypersphere.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        # Using A5-B6 for N-cube as recommended
        self.place_in_area(n_cube, 'A5', 'B6', scale_factor=0.6)
        # Using C5-D6 for N-sphere as recommended
        self.place_in_area(n_sphere, 'C5', 'D6', scale_factor=0.7)
        self.play(Create(n_cube), Create(n_sphere))
        
        # === Animation for Lecture Line 2 ===
        # The equation follows the same sum-of-squares pattern.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Using E5-F6 for formula as recommended
        self.place_in_area(formula, 'E5', 'F6', scale_factor=0.7)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 3 ===
        # We represent an N-dimensional unit sphere algebraically.
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(1)
