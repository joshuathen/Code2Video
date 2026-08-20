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
        self.setup_layout("Visual Synthesis & Conclusion", [
            "Differentiation finds the slope of curves.",
            "Integration calculates the area underneath.",
            "The Fundamental Theorem bridges these concepts."
        ])
        
        # Create elements
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False}).scale(0.5)
        graph = axes.plot(lambda x: 0.2 * x**2 + 1, x_range=[0, 4], color=YELLOW)
        
        # Get point on graph
        x_val = 1.5
        point = axes.c2p(x_val, 0.2 * x_val**2 + 1)
        # Approximate tangent line
        line = Line(start=LEFT, end=RIGHT, color=RED, stroke_width=4).scale(0.5)
        line.move_to(point)
        # Rotate line to match derivative slope at x=1.5
        # f'(x) = 0.4x. f'(1.5) = 0.6. Slope angle = arctan(0.6)
        line.rotate(np.arctan(0.6))
        
        area = axes.get_area(graph, x_range=[0, 3], color=BLUE, opacity=0.3)
        
        # Group scene elements
        right_side = VGroup(axes, graph, line, area)
        # Applying requested improvements (incorporating latest instruction: B2 to F6, 0.85 scale)
        self.place_in_area(right_side, "B2", "F6", scale_factor=0.85)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED)
        self.play(Create(axes), Create(graph))
        self.play(Create(line))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(area))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        bridge = Text("Fundamental Theorem", font_size=24, color=GREEN)
        # Applying requested improvements (A4, 0.9 scale)
        self.place_at_grid(bridge, "A4", scale_factor=0.9)
        self.play(Write(bridge))
        self.wait(2)
