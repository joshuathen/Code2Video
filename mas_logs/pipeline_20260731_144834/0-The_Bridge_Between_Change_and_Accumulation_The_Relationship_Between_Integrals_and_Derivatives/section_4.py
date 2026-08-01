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
        self.setup_layout(
            "FTC Part 1: The Rate of Growth", 
            [
                "Consider a tiny increase in the area, dA.",
                "This narrow strip has a width of dx.",
                "Its height is simply the function's value, f(x).",
                "Thus, the rate of area growth is f(x).",
                "The derivative of the integral is the original function."
            ]
        )

        # Colors
        COLOR_F = "#00FF00"  # Green for f(x)
        COLOR_DX = "#FFFFFF" # White for dx
        COLOR_DA = "#FFFFFF" # White for dA
        COLOR_RESULT = "#FFFFFF" # White for final derivative

        # Setup Axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": BLUE_D},
        )
        self.place_in_area(axes, "B1", "E5", scale_factor=0.85)
        
        # Define function f(x) = 0.1*(x-1)**2 + 1.5
        func = axes.plot(lambda x: 0.1 * (x - 1)**2 + 1.5, x_range=[0, 4.5], color=BLUE_B)
        # B008: Use Text instead of MathTex for labels
        func_label = Text("f(x)", color=BLUE_B, font_size=24)
        self.place_at_grid(func_label, "B5", scale_factor=0.8)

        # Initial area A(x) up to x=2
        x_val = 2.0
        area_initial = axes.get_area(func, x_range=[0.5, x_val], color=BLUE_E, opacity=0.5)
        
        # Vertical bar at x
        scanning_bar = Line(
            axes.c2p(x_val, 0),
            axes.c2p(x_val, 0.1 * (x_val - 1)**2 + 1.5),
            color=WHITE,
            stroke_width=2
        )
        x_label = Text("x", color=WHITE, font_size=24)
        x_label.move_to(axes.c2p(x_val, -0.3))

        self.add(axes, func, func_label, area_initial, scanning_bar, x_label)

        # === Animation for Lecture Line 1 ===
        # Consider a tiny increase in the area, dA.
        self.lecture[0].set_color(COLOR_DA)
        
        dx_val = 0.4
        area_increment = axes.get_area(func, x_range=[x_val, x_val + dx_val], color=YELLOW, opacity=0.7)
        
        self.play(Create(area_increment), run_time=1.5)
        
        da_label = Text("dA", color=COLOR_DA, font_size=24)
        # Position label above the increment area
        da_label.move_to(axes.c2p(x_val + dx_val/2, 0.1 * (x_val - 1)**2 + 1.5 + 0.5))
        
        self.play(Write(da_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This narrow strip has a width of dx.
        self.lecture[1].set_color(COLOR_DX)
        
        dx_brace = BraceBetweenPoints(
            axes.c2p(x_val, 0),
            axes.c2p(x_val + dx_val, 0),
            direction=DOWN,
            color=COLOR_DX,
            buff=0.1
        )
        dx_text = Text("dx", color=COLOR_DX, font_size=24)
        dx_text.next_to(dx_brace, DOWN, buff=0.1)
        
        self.play(Create(dx_brace), Write(dx_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Its height is simply the function's value, f(x).
        self.lecture[2].set_color(COLOR_F)
        
        height_val = 0.1 * (x_val + dx_val - 1)**2 + 1.5
        height_line = Line(
            axes.c2p(x_val + dx_val, 0),
            axes.c2p(x_val + dx_val, height_val),
            color=COLOR_F,
            stroke_width=4
        )
        f_x_label = Text("f(x)", color=COLOR_F, font_size=24)
        f_x_label.next_to(height_line, RIGHT, buff=0.1)
        
        self.play(Create(height_line), Write(f_x_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Thus, the rate of area growth is f(x).
        self.lecture[3].set_color(COLOR_DA)
        
        # Equation dA ≈ f(x) * dx
        eq_slice = Text("dA ≈ f(x) · dx", color=COLOR_DA, font_size=28)
        # Fix for issue 28: Move from A4 to A5 and adjust scale
        self.place_at_grid(eq_slice, "A5", scale_factor=0.8)
        
        self.play(Write(eq_slice))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The derivative of the integral is the original function.
        self.lecture[4].set_color(COLOR_RESULT)
        
        # Final FTC equation
        ftc_eq = Text("d/dx [A(x)] = f(x)", color=COLOR_RESULT, font_size=28)
        # Fix for issue 29: Move from F4 to F5 and adjust scale
        self.place_at_grid(ftc_eq, "F5", scale_factor=0.9)
        
        # Asset: based icon
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(based_icon, "F6", scale_factor=0.6)
        
        self.play(Write(ftc_eq), FadeIn(based_icon))
        self.play(Indicate(ftc_eq, color=COLOR_RESULT, scale_factor=1.1))
        self.wait(2)
