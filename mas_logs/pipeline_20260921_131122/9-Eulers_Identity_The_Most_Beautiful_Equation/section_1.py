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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: The Complex Plane", [
            "The complex plane has real and imaginary axes.",
            "A point (a, b) maps to a + bi.",
            "We can represent this point using distance and angle."
        ])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Axes
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True})
        axes.set_color(WHITE)
        # Fixed per issue 21/36
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.5)
        
        # Point
        a, b = 3, 2
        z_point = Dot(axes.c2p(a, b), color=GREEN)
        
        # Vector
        vector = Line(axes.c2p(0, 0), axes.c2p(a, b), color=YELLOW)
        
        # Labels
        z_label = MathTex("z = a + bi", color=GREEN)
        # Fixed per issue 22/37
        self.place_at_grid(z_label, 'A4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(ruler, 'A2', scale_factor=0.3)
        self.play(FadeIn(ruler), Create(axes))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(Create(vector), Create(z_point))
        self.play(Write(z_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(protractor, 'F2', scale_factor=0.3)
        self.play(FadeIn(protractor))
        
        r_label = MathTex("r = \\sqrt{a^2 + b^2}", color=YELLOW)
        theta_label = MathTex("\\theta = \\arctan(b/a)", color=YELLOW)
        polar_form = MathTex("z = r(\\cos \\theta + i \\sin \\theta)", color=YELLOW)
        
        # Fixed per issue 22/37
        self.place_at_grid(r_label, 'B4', scale_factor=0.7)
        self.place_at_grid(theta_label, 'C4', scale_factor=0.7)
        
        self.play(Write(r_label), Write(theta_label))
        
        # Fixed per issue 23/38
        self.place_in_area(polar_form, 'D4', 'E5', scale_factor=0.6)
        
        self.play(ReplacementTransform(VGroup(z_label, r_label, theta_label), polar_form))
        
        self.wait(2)
