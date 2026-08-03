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
        # Data
        title = "Visual Proof: The Accumulation Function"
        lines = [
            "Define A of x as the accumulated area.",
            "If x moves slightly, area grows by a rectangle.",
            "The rectangle's height is the function's value.",
            "The growth rate is exactly the function's height.",
            "Thus, the derivative of area is the function."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_F = "#FFFFFF"
        COLOR_A = "#00FA9A"
        COLOR_BAR = "#FF0000"
        COLOR_RECT = "#FFD700"

        # Axes Setup
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "B2", "D5", scale_factor=0.6)
        
        f = lambda t: 0.5 * np.sin(t) + 1.5
        graph = axes.plot(f, color=COLOR_F, x_range=[0, 4.5])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        graph_label = MathTex("f(t)", color=COLOR_F)
        self.place_at_grid(graph_label, "A5", scale_factor=0.7)
        
        a_val = 1
        x_val = 2.5
        area_a = axes.get_area(graph, x_range=[a_val, x_val], color=COLOR_A, opacity=0.4)
        area_label = MathTex("A(x)", color=COLOR_A)
        self.place_at_grid(area_label, "C3", scale_factor=0.8)
        
        self.play(Create(axes), Create(graph), Write(graph_label))
        self.play(FadeIn(area_a), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        dx = 0.6
        v_line_x = axes.get_vertical_line(axes.c2p(x_val, f(x_val)), color=COLOR_BAR)
        v_line_xdx = axes.get_vertical_line(axes.c2p(x_val + dx, f(x_val + dx)), color=COLOR_BAR)
        
        # Small labels for x and x+dx
        x_lbl = MathTex("x", color=WHITE)
        self.place_at_grid(x_lbl, "E3", scale_factor=0.6)
        xdx_lbl = MathTex("x+dx", color=WHITE)
        self.place_at_grid(xdx_lbl, "E4", scale_factor=0.6)
        
        rect_da = axes.get_area(graph, x_range=[x_val, x_val + dx], color=COLOR_RECT, opacity=0.7)
        da_lbl = MathTex("dA", color=COLOR_RECT)
        # Issue 31: Fix position of da_lbl to B5
        self.place_at_grid(da_lbl, "B5", scale_factor=0.7)
        
        self.play(Create(v_line_x), FadeIn(x_lbl))
        self.wait(0.5)
        self.play(Create(v_line_xdx), FadeIn(xdx_lbl))
        self.play(FadeIn(rect_da), Write(da_lbl))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        height_brace = BraceBetweenPoints(axes.c2p(x_val, 0), axes.c2p(x_val, f(x_val)), LEFT, color=COLOR_RECT, buff=0.1)
        height_lbl = MathTex("f(x)", color=COLOR_RECT)
        self.place_at_grid(height_lbl, "B4", scale_factor=0.7)
        
        self.play(Create(height_brace), Write(height_lbl))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        formula_da = MathTex(r"dA \approx f(x) \cdot dx", color=COLOR_RECT)
        # Issue 32: Fix position of formula_da to area F2-F3
        self.place_in_area(formula_da, "F2", "F3", scale_factor=0.8)
        
        self.play(Write(formula_da))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_formula = MathTex(r"\frac{dA}{dx} = f(x)", color=COLOR_A)
        # Issue 33: Fix position of final_formula to area F4-F5
        self.place_in_area(final_formula, "F4", "F5", scale_factor=0.8)
        
        self.play(FadeIn(final_formula))
        self.play(Indicate(final_formula))
        self.wait(3)
