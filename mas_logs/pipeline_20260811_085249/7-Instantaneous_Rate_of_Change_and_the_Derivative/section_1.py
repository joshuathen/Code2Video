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
        lecture_lines = [
            "Average rate is the secant slope.",
            "It views the 'big picture'.",
            "Average rate ignores specific behavior."
        ]
        self.setup_layout("The Concept of Average Rate (Prerequisite)", lecture_lines)
        
        # Setup plot area
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1], x_length=4, y_length=4).shift(RIGHT * 1.5)
        curve = axes.plot(lambda x: 0.1 * x**3, color=WHITE)
        
        a, b = 1.0, 4.0
        
        # Using asset
        icon_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=YELLOW)
        icon_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=YELLOW)
        
        # Placing points
        start_point = Dot(axes.c2p(a, 0.1 * a**3), color=YELLOW)
        self.place_at_grid(start_point, 'D3', scale_factor=1.2)
        
        end_point = Dot(axes.c2p(b, 0.1 * b**3), color=YELLOW)
        
        secant = Line(start_point.get_center(), end_point.get_center(), color=BLUE_C)
        
        formula = MathTex(r"\text{Slope} = \frac{f(b)-f(a)}{b-a}", font_size=32, color=WHITE)
        self.place_in_area(formula, 'D4', 'E6', scale_factor=0.9)
        
        slope_label = Text("Slope", font_size=20, color=BLUE_C)
        self.place_at_grid(slope_label, 'E4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve))
        self.lecture[0].set_color(BLUE_C)
        self.play(FadeIn(start_point), FadeIn(end_point), Create(secant))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        self.play(FadeIn(formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.play(Indicate(secant))
        self.wait(2)
