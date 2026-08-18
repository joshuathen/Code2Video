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
        self.setup_layout("Newton's Method: The Iterative Search", [
            "Newton's method uses an iterative formula.",
            "Tangent lines project toward the root.",
            "Each step improves the current approximation."
        ])

        # Define function: f(x) = x^2 - 2 (simple curve)
        axes = Axes(x_range=[-2, 3], y_range=[-2, 4], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2 - 2, color=WHITE)
        
        # Setup visual elements
        # Applying requested changes (Issue 30 and 28): Axes to A4-D6
        axes_group = VGroup(axes, curve)
        self.place_in_area(axes_group, 'A4', 'D6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # Show function and initial guess x_0
        x0 = 2.5
        dot_x0 = Dot(axes.c2p(x0, x0**2 - 2), color="#FFFFFF")
        label_x0 = MathTex("x_0", color="#FFFFFF", font_size=24).next_to(dot_x0, UP)
        self.play(Create(dot_x0), Write(label_x0))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        # Draw tangent line at x_0
        def tangent_at(x_val):
            m = 2 * x_val
            y_val = x_val**2 - 2
            return Line(axes.c2p(x_val-1, y_val-m), axes.c2p(x_val+1, y_val+m), color="#00FFFF")
            
        tangent = tangent_at(x0)
        self.play(Create(tangent))
        
        # Intersection x_1
        x1 = 1.65
        dot_x1 = Dot(axes.c2p(x1, 0), color="#FFFF00")
        label_x1 = MathTex("x_1", color="#FFFF00", font_size=24).next_to(dot_x1, DOWN)
        self.play(Create(dot_x1), Write(label_x1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Highlight repetition
        tangent2 = tangent_at(x1)
        self.play(FadeOut(tangent), Create(tangent2))
        
        # Second intersection x_2
        x2 = 1.43
        dot_x2 = Dot(axes.c2p(x2, 0), color="#00FF00")
        label_x2 = MathTex("x_2", color="#00FF00", font_size=24).next_to(dot_x2, DOWN)
        self.play(Create(dot_x2), Write(label_x2))
        
        # Formula display (Issue 29: Formula to E4)
        formula = MathTex("x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}", color="#FFFFFF")
        self.place_at_grid(formula, 'E4', scale_factor=0.7)
        self.play(Write(formula))
        self.wait(2)
