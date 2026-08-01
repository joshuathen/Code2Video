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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initializing the layout with the title and lecture lines
        # Using unicode \u03c0 for pi to ensure symbol compatibility
        self.setup_layout("The Grand Reunion: e^i\u03c0 + 1 = 0", [
            "Add one to both sides to reach perfect balance.",
            "Growth, rotation, and geometry finally meet at zero.",
            "Euler's identity: the most beautiful equation in all mathematics."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(WHITE))

        # Define the starting equation e^(iπ) = -1
        # Using VGroup of MarkupText for modularity and to avoid LaTeX dependencies
        eq_start = VGroup(
            MarkupText("e<sup>iπ</sup>"),
            MarkupText("="),
            MarkupText("-1")
        ).arrange(RIGHT, buff=0.2).set_color(WHITE)
        
        # Place the starting equation in the designated area (Issue 42 prep)
        self.place_in_area(eq_start, 'A2', 'C5', scale_factor=1.1)
        
        # Define the final identity: e^(iπ) + 1 = 0
        # euler_eq: e^{iπ} + 1 = 0
        euler_eq = VGroup(
            MarkupText("e<sup>iπ</sup>"),
            MarkupText("+"),
            MarkupText("1"),
            MarkupText("="),
            MarkupText("0")
        ).arrange(RIGHT, buff=0.2).set_color(WHITE)
        euler_eq[4].set_color(GOLD) # The '0' (#FFD700) balances the identity
        
        # Ensure the final equation is correctly positioned in the grid (Issue 42)
        self.place_in_area(euler_eq, 'A2', 'C5', scale_factor=1.1)

        self.play(Write(eq_start))
        self.wait(1)
        
        # Animate the transition from -1 to the balanced identity on the left side
        self.play(ReplacementTransform(eq_start, euler_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line - meeting at zero
        self.play(self.lecture[1].animate.set_color(GOLD))

        # Create geometric visualization: Unit circle representing the pi rotation
        circle = Circle(radius=1.0, color=WHITE)
        dot_at_pi = Dot(circle.point_at_angle(PI), color=RED)
        radius_line = Line(circle.get_center(), dot_at_pi.get_center(), color=YELLOW)
        center_dot = Dot(circle.get_center(), color=WHITE)
        unit_circle_viz = VGroup(circle, dot_at_pi, radius_line, center_dot)
        
        # Anchor the geometric visualization to the grid area (Issue 43)
        self.place_in_area(unit_circle_viz, 'D2', 'F5', scale_factor=0.9)
        
        # Label for the meeting point 'zero'
        zero_label = Text("0", color=GOLD)
        # Position the label at a precise grid coordinate (Issue 44)
        self.place_at_grid(zero_label, 'F3', scale_factor=0.8)

        self.play(Create(circle), Create(radius_line))
        self.play(FadeIn(dot_at_pi), FadeIn(center_dot))
        self.play(Write(zero_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Final focus on the full equation representing the beauty of mathematics
        self.play(Indicate(euler_eq))
        self.wait(2)
