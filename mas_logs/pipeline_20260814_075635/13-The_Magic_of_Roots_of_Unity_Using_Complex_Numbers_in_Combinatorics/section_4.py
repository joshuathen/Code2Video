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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Geometric cancellation is powerful.",
            "Complex filters solve combinatorial problems easily.",
            "Visual intuition leads to algebraic solutions."
        ]
        self.setup_layout("Summary and Geometric Intuition", lecture_lines)
        
        # --- Assets ---
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        
        # --- Visual Objects ---
        # 1. Circle representing the complex plane
        circle = Circle(radius=1.5, color=BLUE)
        dot = Dot(circle.point_from_proportion(0), color=WHITE)
        vector = Line(ORIGIN, dot.get_center(), color=WHITE)
        
        # 2. Projection objects
        basis_line = Line(LEFT*2, RIGHT*2, color=GRAY)
        
        # 3. Final Signal
        signal = VGroup(*[Square(side_length=0.3, fill_opacity=0.5, color=WHITE) for _ in range(5)])
        signal.arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF8800"))
        
        # Fix 38: Move circle_animation to B5
        self.place_at_grid(circle, 'B5', scale_factor=0.6)
        
        # Use compass asset
        self.place_at_grid(compass, 'B6', scale_factor=0.5)
        
        self.play(Create(circle), FadeIn(compass))
        
        # Fix 40: Move dot_animation to E2
        self.place_at_grid(dot, 'E2', scale_factor=0.5)
        self.place_at_grid(vector, 'E2', scale_factor=0.5)
        
        self.play(Create(vector), Create(dot))
        # Note: Rotation logic simplified as objects were scaled/moved to grid
        self.play(Rotate(vector, angle=TAU, about_point=self.grid['B5']), Rotate(dot, angle=TAU, about_point=self.grid['B5']), run_time=3)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Fix 39: Move rectangular_grid_group (using signal placeholder)
        self.place_in_area(basis_line, 'A4', 'F6', scale_factor=0.6)
        
        self.play(Create(basis_line))
        
        # Project dot onto basis
        proj_dot = Dot(color="#FFFF00")
        proj_dot.add_updater(lambda m: m.move_to([dot.get_center()[0], basis_line.get_center()[1], 0]))
        self.add(proj_dot)
        self.play(FadeIn(proj_dot))
        self.play(FadeOut(proj_dot), FadeOut(basis_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Use ruler asset
        self.place_at_grid(ruler, 'D3', scale_factor=0.5)
        self.place_at_grid(signal, 'C3', 1.0)
        
        self.play(FadeIn(signal), FadeIn(ruler))
        self.play(signal.animate.set_color("#00FF00"), run_time=2)
        
        self.wait(2)
