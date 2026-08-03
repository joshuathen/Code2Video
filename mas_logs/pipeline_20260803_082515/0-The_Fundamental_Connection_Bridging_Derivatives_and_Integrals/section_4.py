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
        # Fetching data from storyboard
        title_text = "The 'Aha!' Moment: The Rate of Area Growth"
        lecture_lines = [
            "How fast does this area function A(x) change?",
            "Move x by a tiny amount called dx.",
            "This adds a small rectangular sliver of area.",
            "The height of this sliver is exactly f(x).",
            "Thus, the derivative of area is the height."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Function to represent the curve
        def func(x):
            return 0.1 * (x - 1)**2 + 1.2

        # Setup Axes on the right side grid area - Fixed Issue 32
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=4.0,
            y_length=3.0,
            axis_config={"color": BLUE, "include_tip": False},
        )
        self.place_in_area(axes, "B2", "F6")
        
        curve = axes.plot(func, x_range=[0, 4.5], color=BLUE)
        
        # Point of interest
        x_val = 2.2
        dx_val = 0.5
        
        # Main area under curve up to x
        main_area = axes.get_area(curve, x_range=[0, x_val], color=BLUE, opacity=0.3)
        
        # Labels for x and f(x)
        x_label = MathTex("x", color=WHITE, font_size=24).next_to(axes.c2p(x_val, 0), DOWN)
        f_label = MathTex("f(x)", color=BLUE, font_size=24).next_to(curve.point_from_proportion(0.6), UP)

        # === Animation for Lecture Line 1 ===
        # "How fast does this area function A(x) change?"
        self.lecture[0].set_color(WHITE)
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(main_area), Write(x_label), Write(f_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Move x by a tiny amount called dx."
        self.lecture[1].set_color("#FFD700") # Gold to match dx
        
        # Brace for dx on x-axis
        dx_brace = BraceBetweenPoints(axes.c2p(x_val, 0), axes.c2p(x_val + dx_val, 0), color="#FFD700", buff=0.1)
        dx_text = MathTex("dx", color="#FFD700", font_size=20).next_to(dx_brace, DOWN, buff=0.05)
        
        self.play(GrowFromCenter(dx_brace), Write(dx_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This adds a small rectangular sliver of area."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700") # Gold to match sliver
        
        # Sliver area represented as a rectangle to illustrate growth
        sliver_rect = Rectangle(
            width=axes.c2p(dx_val, 0)[0] - axes.c2p(0, 0)[0],
            height=axes.c2p(0, func(x_val))[1] - axes.c2p(0, 0)[1],
            fill_color="#FFD700",
            fill_opacity=0.6,
            stroke_width=1,
            stroke_color="#FFD700"
        )
        sliver_rect.move_to(axes.c2p(x_val + dx_val/2, func(x_val)/2))
        
        da_label = MathTex("dA", color="#FFD700", font_size=24).move_to(axes.c2p(x_val + dx_val/2, func(x_val) + 0.3))
        
        self.play(FadeIn(sliver_rect), Write(da_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The height of this sliver is exactly f(x)."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF00FF") # Pink to match height arrow
        
        # Vertical pink arrow for height
        height_arrow = DoubleArrow(
            axes.c2p(x_val + dx_val, 0), 
            axes.c2p(x_val + dx_val, func(x_val)), 
            color="#FF00FF", 
            buff=0,
            stroke_width=3
        )
        h_label = MathTex("f(x)", color="#FF00FF", font_size=24).next_to(height_arrow, RIGHT, buff=0.1)
        
        self.play(Create(height_arrow), Write(h_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Thus, the derivative of area is the height."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF") # White for equation
        
        # Fundamental equation - Fixed Issue 31
        ftc_eq = MathTex(r"\frac{dA}{dx} = f(x)", color=WHITE, font_size=42)
        self.place_in_area(ftc_eq, "A3", "A4", scale_factor=0.8)
        
        self.play(Write(ftc_eq))
        self.wait(2)
