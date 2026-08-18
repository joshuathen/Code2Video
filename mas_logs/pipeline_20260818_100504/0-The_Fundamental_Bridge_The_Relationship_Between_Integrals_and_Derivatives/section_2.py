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
        lecture_lines = ["Derivatives act like a zoom lens.", "They reveal slope at any point.", "Imagine the rocket's vertical velocity."]
        self.setup_layout("Visualizing the Derivative: The Slope Inspector", lecture_lines)
        
        # Setup Axes
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        
        # Curve
        curve = axes.plot(lambda x: 0.2 * x**3, color=WHITE)
        
        # Points and Secant
        p1 = ValueTracker(1.0)
        p2 = ValueTracker(3.0)
        
        def get_point(t): return axes.c2p(t, 0.2 * t**3)
        
        dot1 = Dot(get_point(p1.get_value()), color=YELLOW)
        dot2 = Dot(get_point(p2.get_value()), color=YELLOW)
        
        secant = always_redraw(lambda: Line(get_point(p1.get_value()), get_point(p2.get_value()), color=YELLOW))
        
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg")
        self.place_at_grid(rocket, 'A4', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve), FadeIn(rocket))
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(dot1), FadeIn(dot2))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(Create(secant))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        
        # Shrink the gap
        dot2.add_updater(lambda d: d.move_to(axes.c2p(p2.get_value(), 0.2 * p2.get_value()**3)))
        rocket.add_updater(lambda r: r.move_to(axes.c2p(p2.get_value(), 0.2 * p2.get_value()**3)))
        
        self.play(p2.animate.set_value(1.1), run_time=3)
        self.play(FadeOut(dot2), FadeOut(secant), FadeOut(rocket))
