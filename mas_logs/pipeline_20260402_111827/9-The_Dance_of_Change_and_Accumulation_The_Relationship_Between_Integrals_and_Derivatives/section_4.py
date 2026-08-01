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
        # Setup Title and Lecture Lines
        title_text = "The Grand Connection: Two Sides of the Same Coin"
        lecture_lines = [
            "Integration and differentiation are actually inverse operations.",
            "Derivatives break paths down into instantaneous speeds.",
            "Integrals build those speeds back into total paths.",
            "They are two sides of the same mathematical coin."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        pink_color = "#FF69B4"
        green_color = "#32CD32"
        formula_color = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create boxes
        diff_box = Rectangle(width=1.8, height=1.2, color=pink_color, fill_opacity=0.2)
        diff_label = Text("Differentiate", font_size=18, color=pink_color)
        diff_group = VGroup(diff_box, diff_label)
        # Fix Issue 36: Move diff_group to B2 and scale to 0.8
        self.place_at_grid(diff_group, "B2", scale_factor=0.8)
        
        int_box = Rectangle(width=1.8, height=1.2, color=green_color, fill_opacity=0.2)
        int_label = Text("Integrate", font_size=18, color=green_color)
        int_group = VGroup(int_box, int_label)
        # Fix Issue 37: Move int_group to B5 and scale to 0.8
        self.place_at_grid(int_group, "B5", scale_factor=0.8)

        self.play(Create(diff_group), Create(int_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(pink_color)

        # Formula f(x)
        fx = Text("f(x)", color=formula_color, font_size=32)
        # Fix Issue 35: Move fx to C2 and scale to 0.8 to avoid notes
        self.place_at_grid(fx, "C2", scale_factor=0.8)
        
        # Emerging derivative f'(x)
        dfx = Text("f'(x)", color=formula_color, font_size=32)
        # Positioned in center row B between boxes
        mid_point = (self.grid["B3"] + self.grid["B4"]) / 2
        dfx.move_to(mid_point)
        
        self.play(FadeIn(fx))
        self.wait(0.5)
        
        # Move f(x) into pink box and emerge as f'(x)
        self.play(fx.animate.move_to(diff_group.get_center()), run_time=1)
        self.play(FadeOut(fx, shift=RIGHT), FadeIn(dfx, shift=RIGHT))
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(green_color)

        # Original result f(x) + C
        fx_c = Text("f(x) + C", color=formula_color, font_size=32)
        # Position at C5 (emerging downwards from int_group)
        fx_c.move_to(self.grid["C5"])
        
        # Move f'(x) into green box and emerge as f(x) + C
        self.play(dfx.animate.move_to(int_group.get_center()), run_time=1)
        self.play(FadeOut(dfx, shift=RIGHT), FadeIn(fx_c, shift=RIGHT))
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Show connection: Curved arrow from end back to start
        # Use row A to clear the boxes on row B
        arrow = CurvedArrow(
            start_point=self.grid["A5"] + UP*0.2, 
            end_point=self.grid["A2"] + UP*0.2, 
            angle=-TAU/4, 
            color=WHITE
        )
        
        self.play(Create(arrow))
        self.wait(2)
