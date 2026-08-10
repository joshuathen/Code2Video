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
        lecture_lines = ["The derivative measures the instantaneous rate of change.", 
                         "Visualize this as a speedometer at one frozen moment.", 
                         "It captures the slope of f(x) at point x."]
        self.setup_layout("Prerequisite Review: The Rate of Change", lecture_lines)
        
        # Elements
        rate_symbol = MathTex(r"dR/dt", color=WHITE)
        axes = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=3).scale(0.6)
        curve = axes.plot(lambda x: 0.5 * x**2, color=BLUE)
        point = Dot(color=RED)
        tangent = TangentLine(curve, alpha=0.5, length=1, color=YELLOW)
        triangle = Polygon(
            axes.c2p(1, 0.5), axes.c2p(2, 0.5), axes.c2p(2, 2),
            color=GREEN, stroke_width=2
        )
        rise = MathTex("rise", color="#FFCC00").scale(0.5)
        run = MathTex("run", color="#FFCC00").scale(0.5)
        
        # Asset for speedometer
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        
        # VGroup for graph area
        graph = VGroup(axes, curve, point, tangent, triangle)
        # Applying critique fixes for graph placement
        self.place_in_area(graph, 'C3', 'F5', scale_factor=0.65)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(rate_symbol))
        # Applying critique fix for rate_symbol placement
        self.place_at_grid(rate_symbol, 'B4', scale_factor=0.9)
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(point), Create(tangent))
        # Integrate speedometer asset
        self.place_at_grid(speedometer, 'B2', scale_factor=0.5)
        self.play(FadeIn(speedometer))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Create(triangle))
        rise.next_to(triangle, RIGHT, buff=0.1)
        run.next_to(triangle, DOWN, buff=0.1)
        self.play(Write(rise), Write(run))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
