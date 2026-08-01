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
        title = "The Big Reveal: The Fundamental Theorem"
        lines = [
            "Look closely at a tiny sliver of area.",
            "Its height is the original function's value.",
            "Its width is a tiny change in x.",
            "The rate of area growth equals the height.",
            "The derivative 'undoes' the integral."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"
        AREA_COLOR = "#4169E1"
        SLIVER_COLOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Zoom into the right edge of the blue area from the previous section.
        # Section 3 used f(t) = 2t, area up to x=3.
        # Use axes that match the previous section's scale for continuity
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 8, 2],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": WHITE}
        ).add_coordinates()
        self.place_in_area(axes, "B1", "E6")
        
        # Consistent function f(x) = 2x (Issue 36)
        func = lambda x: 2 * x
        graph = axes.plot(func, x_range=[0, 3.8], color=WHITE)
        area = axes.get_area(graph, x_range=[0, 3.0], color=AREA_COLOR, opacity=0.5)
        
        self.add(axes, graph, area)
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Isolate a sliver at x=3.0
        sliver_x = 3.0
        dx_val = 0.2
        sliver = axes.get_area(graph, x_range=[sliver_x, sliver_x + dx_val], color=SLIVER_COLOR, opacity=0.8)
        
        # === Animation for Lecture Line 2 ===
        # Isolate a thin vertical sliver of width dx and height f(x).
        self.play(self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Create(sliver))
        
        # Label height f(x)
        # Issue 27: self.place_at_grid(height_label, 'C4', scale_factor=0.7)
        h_line = Line(axes.c2p(sliver_x, 0), axes.c2p(sliver_x, func(sliver_x)), color=SLIVER_COLOR, stroke_width=4)
        height_label = MathTex("f(x)", color=WHITE)
        self.place_at_grid(height_label, 'C4', scale_factor=0.7)
        
        self.play(Create(h_line), Write(height_label))
        
        # === Animation for Lecture Line 3 ===
        # Label the area of the sliver as dA = f(x) * dx.
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Brace for width dx
        width_brace = Brace(sliver, DOWN, buff=0.1, color=WHITE)
        dx_label = MathTex("dx", color=WHITE)
        dx_label.next_to(width_brace, DOWN, buff=0.1).scale(0.7)
        
        # Issue 28: self.place_at_grid(da_label, 'A5', scale_factor=0.8)
        da_label = MathTex("dA = f(x) \\cdot dx", color=SLIVER_COLOR)
        self.place_at_grid(da_label, 'A5', scale_factor=0.8)
        
        self.play(Create(width_brace), Write(dx_label))
        self.wait(0.5)
        self.play(Write(da_label))
        
        # === Animation for Lecture Line 4 ===
        # Animate the algebraic rearrangement to dA/dx = f(x) using a spinning gear icon.
        # Issue 20: Use SVGMobject for gear.
        # Issue 28: self.place_at_grid(da_dx_eq, 'B5', scale_factor=1.0)
        self.play(self.lecture[3].animate.set_color(HIGHLIGHT_COLOR))
        
        da_dx_eq = MathTex(r"\frac{dA}{dx} = f(x)", color=SLIVER_COLOR)
        self.place_at_grid(da_dx_eq, 'B5', scale_factor=1.0)
        
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg", color=GREY_B)
        self.place_at_grid(gear, "C5", scale_factor=0.5)
        
        self.play(
            FadeIn(gear),
            Transform(da_label.copy(), da_dx_eq)
        )
        
        # Spin the gear while showing the result
        self.play(Rotate(gear, angle=2*PI, run_time=2, rate_func=linear))
        
        # === Animation for Lecture Line 5 ===
        # The text 'Fundamental Theorem' appears and flashes in gold (#FFD700).
        # Issue 26: self.place_in_area(ftc_text, 'F1', 'F6', scale_factor=1.0)
        self.play(self.lecture[4].animate.set_color(HIGHLIGHT_COLOR))
        
        ftc_text = Text("Fundamental Theorem", color="#FFD700")
        self.place_in_area(ftc_text, 'F1', 'F6', scale_factor=1.0)
        
        self.play(Write(ftc_text))
        
        # Flash effect
        for _ in range(2):
            self.play(ftc_text.animate.set_color(WHITE), run_time=0.25)
            self.play(ftc_text.animate.set_color("#FFD700"), run_time=0.25)
            
        self.wait(2)
