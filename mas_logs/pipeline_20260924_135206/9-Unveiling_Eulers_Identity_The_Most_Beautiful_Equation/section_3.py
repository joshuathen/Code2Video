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
        lecture_lines = [
            "Euler's formula: e^(iθ) = cos(θ) + i sin(θ).",
            "Angle θ sweeps an arc on the circle.",
            "Tracing coordinates as cos and sin.",
            "Input θ links circles to exponentials.",
            "Observe the circle's rotation."
        ]
        self.setup_layout("Euler’s Formula: Linking Circles and Exponentials", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"e^{i\theta} = \cos(\theta) + i \sin(\theta)", font_size=40)
        self.place_in_area(formula, 'B2', 'B6', scale_factor=0.8)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        self.place_at_grid(protractor, 'A5', scale_factor=0.3)
        self.play(Write(formula), FadeIn(protractor))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 2 ===
        axes = Axes(x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], x_length=3, y_length=3)
        circle = Circle(radius=1.5, color=WHITE)
        axes_group = VGroup(axes, circle)
        self.place_at_grid(axes_group, 'D5', scale_factor=0.6)
        
        self.play(Create(axes), Create(circle))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        theta = ValueTracker(0)
        dot = Dot(color="#FF0000")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        def update_dot(d):
            angle = theta.get_value()
            d.move_to(axes.c2p(np.cos(angle), np.sin(angle)))
        
        dot.add_updater(update_dot)
        self.add(dot)
        self.place_at_grid(compass, 'D5', scale_factor=0.3)
        
        self.play(theta.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.play(self.lecture[2].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 4 ===
        rect = SurroundingRectangle(formula, color=BLUE)
        self.play(Create(rect))
        self.play(self.lecture[3].animate.set_color(BLUE))

        # === Animation for Lecture Line 5 ===
        final_text = Text("Rotation completed!", color=YELLOW, font_size=24)
        self.place_at_grid(final_text, 'E5', scale_factor=0.9)
        self.play(FadeIn(final_text))
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.wait(1)
