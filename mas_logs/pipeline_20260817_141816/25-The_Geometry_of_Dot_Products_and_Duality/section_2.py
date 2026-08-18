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
        self.setup_layout("Visualizing Projections", [
            "Geometrically, dot product equals |a||b|cos(θ).",
            "Vector 'a' casts a shadow onto 'b'.",
            "This projects the component onto movement."
        ])
        
        # Define base vectors and elements
        a = Vector([1.5, 1, 0], color=WHITE)
        b = Vector([2.5, 0, 0], color=WHITE)
        
        line_l = Line(start=np.array([-1, 0, 0]), end=np.array([3, 0, 0]), color=WHITE)
        
        # Sunlight asset
        sunlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sunlight.svg")
        
        # Setup mobjects for animation
        # Projection of 'a' onto b
        proj_val = 1.5
        proj_line = Line(start=[0, 0, 0], end=[proj_val, 0, 0], color="#0000FF")
        proj_label = MathTex(r"proj_{\vec{b}}(\vec{a})", font_size=24, color="#0000FF")

        # Position elements based on requirements
        self.place_at_grid(a, 'B3', scale_factor=0.9)
        self.place_at_grid(b, 'B5', scale_factor=0.9)
        self.place_at_grid(line_l, 'C4', scale_factor=0.8)
        self.place_at_grid(sunlight, 'A4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(a), Create(b), Create(line_l), FadeIn(sunlight))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Create(proj_line))
        self.place_at_grid(proj_label, 'E3', scale_factor=0.7)
        self.play(FadeIn(proj_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Indicate(proj_line))
        self.wait(2)
