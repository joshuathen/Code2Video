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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            "As points multiply, they form a continuous glowing ring.",
            "The total brightness approaches a specific geometric limit.",
            "Suddenly, the circle's constant, pi, emerges in the sum.",
            "The result is exactly pi squared over six.",
            "Euler's solution connects integers to the geometry of circles."
        ]
        self.setup_layout("The Limit and the Reveal", lines)

        # Colors
        color1 = "#00FFFF" # Cyan
        color2 = "#FFB347" # Light Orange
        color3 = "#E0B0FF" # Light Purple
        color4 = "#FFFF99" # Light Yellow
        color5 = "#FFD700" # Gold

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color1))
        
        # Discrete points to continuous ring
        points_group = VGroup()
        for i in range(12):
            dot = Dot(radius=0.08, color=color1)
            angle = i * (2 * PI / 12)
            dot.move_to(self.grid["C4"] + np.array([np.cos(angle), np.sin(angle), 0]) * 1.5)
            points_group.add(dot)
        
        self.place_in_area(points_group, "A3", "D5")
        self.play(FadeIn(points_group))
        
        # Increase density
        dense_points = VGroup()
        for i in range(60):
            dot = Dot(radius=0.03, color=color1)
            angle = i * (2 * PI / 60)
            dot.move_to(self.grid["C4"] + np.array([np.cos(angle), np.sin(angle), 0]) * 1.5)
            dense_points.add(dot)
        self.place_in_area(dense_points, "A3", "D5")

        continuous_ring = Circle(radius=1.5, color=color1).set_stroke(width=6)
        self.place_in_area(continuous_ring, "A3", "D5")
        
        self.play(ReplacementTransform(points_group, dense_points))
        self.wait(0.5)
        self.play(ReplacementTransform(dense_points, continuous_ring))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color2))
        
        brightness_label = Text("Total Brightness:", font_size=20, color=color2)
        # Issue 40 fix
        self.place_in_area(brightness_label, "E1", "E2", scale_factor=0.8)
        
        limit_expr = Text("Σ 1/n² → L", font_size=24, color=color2)
        # Issue 41 fix
        self.place_in_area(limit_expr, "E4", "E6", scale_factor=1.0)
        
        self.play(Write(brightness_label), Write(limit_expr))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Highlight geometry relation
        radius_line = Line(continuous_ring.get_center(), continuous_ring.get_right(), color=color3)
        radius_label = Text("R = 1", font_size=18, color=color3).next_to(radius_line, UP, buff=0.1)
        pi_symbol = Text("π", font_size=48, color=color3)
        self.place_at_grid(pi_symbol, "C4", scale_factor=1.5)
        
        self.play(Create(radius_line), FadeIn(radius_label))
        self.play(FadeIn(pi_symbol))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color4))
        
        final_result = Text("Σ 1/n² = π²/6", font_size=28, color=color4)
        # Issue 41 fix
        self.place_in_area(final_result, "E4", "E6", scale_factor=1.1)
        
        self.play(
            FadeOut(limit_expr),
            FadeOut(brightness_label),
            FadeIn(final_result)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color5))
        
        # Euler's connection - Reveal center glow
        euler_name = Text("Euler's Solution", font_size=24, color=color5)
        # Issue 39 fix
        self.place_in_area(euler_name, "E3", "E5", scale_factor=1.0)
        
        # Center glow version
        central_result = final_result.copy().set_color(color5)
        self.place_in_area(central_result, "B2", "D5", scale_factor=1.5)
        
        self.play(
            FadeOut(continuous_ring),
            FadeOut(radius_line),
            FadeOut(radius_label),
            FadeOut(pi_symbol),
            FadeIn(euler_name),
            ReplacementTransform(final_result, central_result)
        )
        
        # Glowing effect
        self.play(central_result.animate.set_stroke(width=2).scale(1.1), rate_func=there_and_back)
        glow_box = SurroundingRectangle(central_result, color=color5, buff=0.2)
        self.play(Create(glow_box))
        self.wait(2)
