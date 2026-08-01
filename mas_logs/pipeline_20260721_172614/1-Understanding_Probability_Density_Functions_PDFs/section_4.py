from manim import *
import numpy as np

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
        self.setup_layout("Mathematical Integration (The Tool)", [
            "Calculus provides the tool to find these shaded areas.",
            "The integral symbol represents the sum of infinitesimal slices.",
            "Integrating the PDF from A to B yields the probability."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Matching color: Purple (#9b59b6) for the integral symbol and tool icon
        self.lecture[0].set_color("#9b59b6")
        
        big_integral = MathTex(r"\int", color="#9b59b6", font_size=180)
        # Issue 36: Place big_integral in B1-D2
        self.place_in_area(big_integral, 'B1', 'D2', scale_factor=0.9)
        
        # Issue 26: Asset integration [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tool.svg]
        tool_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tool.svg")
        tool_icon.set_color("#9b59b6")
        # Place tool icon alongside the integral symbol
        self.place_in_area(tool_icon, 'B3', 'D4', scale_factor=0.6)
        
        self.play(Write(big_integral), FadeIn(tool_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: Yellow (#f1c40f) for the concept of "slices" (Riemann rectangles)
        self.lecture[1].set_color("#f1c40f")
        
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2, 1],
            x_length=4,
            y_length=2.5,
            axis_config={"include_tip": False, "font_size": 20}
        ).set_color(GRAY)
        # Issue 37: Place axes in C3-F6
        self.place_in_area(axes, 'C3', 'F6', scale_factor=0.7)
        
        curve = axes.plot(lambda x: 0.1 * x * (4-x) + 0.5, color=WHITE)
        rects = axes.get_riemann_rectangles(
            curve, x_range=[1.0, 3.0], dx=0.2, 
            stroke_width=0.5, fill_opacity=0.6, color="#f1c40f"
        )
        
        # Fade out the tool icon to clear space for the axes
        self.play(
            Create(axes),
            Create(curve),
            FadeOut(tool_icon),
            run_time=1.5
        )
        self.play(Create(rects))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color: Green (#2ecc71) for the area and the probability bounds
        self.lecture[2].set_color("#2ecc71")
        
        # Solid area representing the probability
        solid_area = axes.get_area(curve, [1.0, 3.0], color="#2ecc71", opacity=0.5)
        
        # Vertical boundary lines at a and b
        line_a = axes.get_vertical_line(axes.c2p(1, curve.underlying_function(1)), color="#2ecc71")
        line_b = axes.get_vertical_line(axes.c2p(3, curve.underlying_function(3)), color="#2ecc71")
        label_a = MathTex("a", color="#2ecc71", font_size=24).next_to(line_a, DOWN, buff=0.1)
        label_b = MathTex("b", color="#2ecc71", font_size=24).next_to(line_b, DOWN, buff=0.1)

        # Formal integration formula
        formula = MathTex(
            r"P(a \le X \le b) = \int_a^b f(x) dx",
            font_size=36
        )
        # Apply matching colors to key elements of the formula
        formula.set_color_by_tex("a", "#2ecc71")
        formula.set_color_by_tex("b", "#2ecc71")
        formula.set_color_by_tex(r"\int", "#9b59b6")
        
        # Issue 35: Place formula in A1-A6
        self.place_in_area(formula, 'A1', 'A6', scale_factor=0.8)

        # Transition slices to solid area and show the formal formula
        self.play(
            FadeOut(big_integral),
            FadeOut(rects),
            FadeIn(solid_area),
            Create(line_a), 
            Create(line_b),
            Write(label_a), 
            Write(label_b),
            Write(formula),
            run_time=2
        )
        
        self.wait(3)
