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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Deriving the Pure Cosine Series"
        lines = [
            "For even functions, all sine terms disappear completely.",
            "This creates a pure cosine series for the function.",
            "The constant average value, a0, remains an even component."
        ]
        self.setup_layout(title, lines)

        # Colors
        filter_color = "#FFFFFF"
        bn_color = "#FF0000"
        a0_color = "#FFFF00"
        cosine_color = "#00FFFF" 

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(filter_color))

        # Create Sine Filter Box (B3 to D4 area)
        filter_box = Rectangle(width=2.0, height=2.0, color=filter_color)
        filter_text = Text("Sine Filter", font_size=20, color=filter_color)
        sine_filter = VGroup(filter_box, filter_text).arrange(UP, buff=0.1)
        self.place_in_area(sine_filter, "B3", "D4")
        
        # Create Even Triangle Wave
        triangle_wave = VMobject(color=WHITE)
        triangle_wave.set_points_as_corners([
            [-0.4, -0.4, 0], [0, 0.4, 0], [0.4, -0.4, 0]
        ])
        # Start at C2 (Maintaining Column 2 gap as per B021)
        self.place_at_grid(triangle_wave, "C2")

        self.play(Create(filter_box), Write(filter_text))
        self.play(Create(triangle_wave))
        self.wait(0.5)

        # Move triangle wave into filter center
        self.play(
            triangle_wave.animate.move_to(sine_filter.get_center()).scale(0.5).set_opacity(0),
            run_time=1.5
        )

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(self.lecture[1].animate.set_color(bn_color))

        # Sine terms (bn) ejected from the filter
        sine_icon = FunctionGraph(lambda x: 0.25 * np.sin(2 * PI * x), x_range=[-0.4, 0.4], color=bn_color)
        bn_label = MathTex("b_n = 0", color=bn_color).scale(0.8)
        
        # Start sine_icon inside filter then move to E4 (Issue 40)
        sine_icon.move_to(sine_filter.get_center())
        # Position bn_label at F4 (Issue 41)
        self.place_at_grid(bn_label, "F4")

        self.play(
            sine_icon.animate.move_to(self.grid["E4"]),
            FadeIn(bn_label),
            run_time=1.5
        )
        self.wait(0.5)
        self.play(FadeOut(sine_icon))

        # Cosine terms emerge from the right side of the filter (C5)
        cosine_icon = FunctionGraph(lambda x: 0.25 * np.cos(2 * PI * x), x_range=[-0.4, 0.4], color=cosine_color)
        self.place_at_grid(cosine_icon, "C5")
        
        self.play(
            FadeIn(cosine_icon, shift=RIGHT),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(self.lecture[2].animate.set_color(a0_color))

        # a0 remains with the cosine terms
        a0_label = MathTex("a_0", color=a0_color).scale(0.8)
        a0_line = Line(start=[-0.3, 0, 0], end=[0.3, 0, 0], color=a0_color)
        a0_group = VGroup(a0_line, a0_label).arrange(UP, buff=0.1)
        
        # Place at D5 (below cosine icon at C5)
        self.place_at_grid(a0_group, "D5")

        self.play(
            FadeIn(a0_group, shift=UP),
            run_time=1
        )
        
        # Group result for final highlight
        result_group = VGroup(cosine_icon, a0_group)
        result_rect = SurroundingRectangle(result_group, color=WHITE, buff=0.2)
        pure_label = Text("Pure Cosine Series", font_size=18, color=WHITE).next_to(result_rect, DOWN)
        
        self.play(Create(result_rect), Write(pure_label))
        self.wait(2)
