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
        # Data from storyboard
        title_text = "Prerequisite Review: Slope and Average Rate"
        lecture_lines = [
            "To find average speed, we use two points.",
            "The secant line connects these points on a graph.",
            "Its slope represents the average rate of change."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#0000FF"
        WHITE_COLOR = "#FFFFFF"

        # Assets
        odometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/od.svg", color=GREEN_COLOR)
        self.place_at_grid(odometer, "B6", scale_factor=0.6)
        
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color=WHITE_COLOR)
        self.place_at_grid(speedometer, "A2", scale_factor=0.5)

        # Axes (Issue 26: place_in_area B1 to F6)
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4.5,
            y_length=4.5
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=0.8)
        
        # Position curve: f(x) = 0.25x^2
        curve = axes.plot(lambda x: 0.25 * x**2, x_range=[0, 4.5], color=WHITE)
        
        # Two points A and B
        x_a, x_b = 1.0, 3.5
        p_a = axes.c2p(x_a, 0.25 * x_a**2)
        p_b = axes.c2p(x_b, 0.25 * x_b**2)
        
        dot_a = Dot(p_a, color=GREEN_COLOR)
        dot_b = Dot(p_b, color=GREEN_COLOR)
        label_a = MathTex("A", color=GREEN_COLOR, font_size=24).next_to(dot_a, LEFT, buff=0.1)
        label_b = MathTex("B", color=GREEN_COLOR, font_size=24).next_to(dot_b, RIGHT, buff=0.1)
        
        # Secant Line
        secant_line = Line(p_a, p_b, color=BLUE_COLOR)
        
        # Slope Formula (Issue 27: place_in_area A3 to A6)
        formula = MathTex(
            r"\text{Slope} = \frac{f(b) - f(a)}{b - a}", 
            color=WHITE_COLOR, 
            font_size=24
        )
        self.place_in_area(formula, "A3", "A6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "To find average speed, we use two points."
        self.play(self.lecture[0].animate.set_color(GREEN_COLOR))
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(dot_a, dot_b), Write(label_a), Write(label_b), FadeIn(odometer))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The secant line connects these points on a graph."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_COLOR)
        )
        self.play(Create(secant_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Its slope represents the average rate of change."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE_COLOR)
        )
        self.play(Write(formula), FadeIn(speedometer))
        self.wait(2)
        
        # Final wait
        self.wait(1)
