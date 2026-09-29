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
            "Euler's formula links exponentials to trigonometry.",
            "The formula is e to the ix equals cos x plus i sin x.",
            "Think of 'x' as an angle on a circle.",
            "The result maps points on the unit circle.",
            "Algebra and geometry meet here beautifully."
        ]
        self.setup_layout("Euler’s Formula: The Geometric Key", lecture_lines)
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Complex plane elements
        axes = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False}).scale(0.4)
        circle = Circle(radius=0.6, color=WHITE)
        point = Dot(color="#FF00FF")
        vector = Arrow(ORIGIN, point.get_center(), color="#FF00FF", buff=0)
        
        # Setup right side (per Critic)
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        circle_animation = VGroup(circle, axes)
        self.place_in_area(circle_animation, 'C3', 'E4', scale_factor=0.5)
        
        point.move_to(axes.c2p(1, 0))
        vector.put_start_and_end_on(axes.get_center(), point.get_center())
        
        # VGroup for manipulation
        complex_group = VGroup(axes, circle, vector, point)
        self.add(complex_group)

        # Asset placement
        self.place_at_grid(compass, "B2", scale_factor=0.3)
        self.add(compass)

        # === Animation for Lecture Line 1 (and 2) ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"), self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 (and 4) ===
        # Rotate point
        angle = ValueTracker(0)
        
        # Add protractor for rotation stage
        self.place_at_grid(protractor, "B5", scale_factor=0.3)
        self.play(FadeIn(protractor))

        vector.add_updater(lambda m: m.put_start_and_end_on(axes.get_center(), axes.c2p(np.cos(angle.get_value()), np.sin(angle.get_value()))))
        point.add_updater(lambda m: m.move_to(axes.c2p(np.cos(angle.get_value()), np.sin(angle.get_value()))))
        
        self.play(self.lecture[2].animate.set_color("#FF00FF"), self.lecture[3].animate.set_color("#FF00FF"))
        self.play(angle.animate.set_value(PI/3), run_time=2)

        # === Animation for Lecture Line 5 ===
        formula = MathTex(r"e^{ix} = \cos(x) + i\sin(x)", color="#00FF00").scale(0.8)
        self.place_at_grid(formula, "B4", scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        self.wait(2)
