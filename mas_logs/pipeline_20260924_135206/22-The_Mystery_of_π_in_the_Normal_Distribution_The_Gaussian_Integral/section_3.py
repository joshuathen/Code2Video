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
        self.setup_layout("The Transformation: Polar Coordinates", [
            "Square the integral to get a 2D surface.",
            "Transition from Cartesian to polar coordinates for simplification.",
            "[Asset: grid_to_ripples_transition] show the geometric transformation."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Square the integral to get a 2D surface.
        self.play(self.lecture[0].animate.set_color(BLUE))
        eqn = MathTex("I^2 = \\iint e^{-(x^2+y^2)} \\, dx \\, dy", color=WHITE)
        self.place_in_area(eqn, 'B2', 'B5', scale_factor=1.0)
        self.play(Write(eqn))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition from Cartesian to polar coordinates for simplification.
        self.play(self.lecture[1].animate.set_color(GREEN))
        polar_eqn = MathTex("x^2 + y^2 = r^2, \\quad dx \\, dy = r \\, dr \\, d\\theta", color=YELLOW)
        self.place_in_area(polar_eqn, 'C2', 'C5', scale_factor=0.9)
        self.play(FadeIn(polar_eqn))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # [Asset: grid_to_ripples_transition] show the geometric transformation.
        self.play(self.lecture[2].animate.set_color(PURPLE))
        
        # Cartesian grid using asset
        grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid.set_color("#888888")
        self.place_in_area(grid, 'D2', 'F5', scale_factor=0.8)
        self.play(FadeIn(grid))
        
        # Point (x, y)
        point = Dot(color=WHITE)
        point.move_to(grid.get_center())
        label = Text("(x, y)", font_size=16, color=WHITE).next_to(point, UP, buff=0.1)
        self.play(Create(point), Write(label))
        
        # Convert to Polar (r, theta)
        new_point = Dot(color=PURPLE)
        new_point.move_to(grid.get_center())
        new_label = Text("(r, \\theta)", font_size=16, color=PURPLE).next_to(new_point, UP, buff=0.1)
        self.play(Transform(point, new_point), Transform(label, new_label))
        self.wait(2)
