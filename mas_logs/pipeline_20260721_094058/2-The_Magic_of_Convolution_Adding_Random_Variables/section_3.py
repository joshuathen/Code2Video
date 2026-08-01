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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The discrete convolution formula defines this sum.",
            "We sum P(X=k) times P(Y=z-k).",
            "This is known as the 'Flip and Slide' method.",
            "Flip one distribution and slide it across another.",
            "Overlapping values are multiplied and then summed together."
        ]
        self.setup_layout("Defining Convolution: Flip and Slide", lecture_lines)

        # Colors
        COLOR_X = BLUE_B
        COLOR_Y = "#FFB6C1" # Light Pink
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        formula = MathTex(
            r"P(Z=z) = \sum_{k} P(X=k) \cdot P(Y=z-k)",
            font_size=36, color=WHITE
        )
        self.place_in_area(formula, 'A1', 'A6')
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        # Highlight the product and sum part
        rhs_box = SurroundingRectangle(formula[0][7:], color=WHITE, buff=0.1)
        self.play(Create(rhs_box))
        self.wait(1)
        self.play(FadeOut(rhs_box))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        def create_chart(probs, color, label_text):
            bars = VGroup()
            for i, p in enumerate(probs):
                rect = Rectangle(
                    width=0.4, height=p*3, 
                    fill_opacity=0.6, fill_color=color, stroke_color=WHITE
                )
                rect.move_to(RIGHT * i * 0.5)
                bars.add(rect)
            
            label = Text(label_text, font_size=20, color=color)
            label.next_to(bars, UP, buff=0.2)
            return VGroup(bars, label)

        chart_x = create_chart([0.3, 0.5, 0.2], COLOR_X, "P(X)")
        chart_y = create_chart([0.4, 0.6], COLOR_Y, "P(Y)")

        # Fix issues 22 and 23 by adjusting positions
        self.place_at_grid(chart_x, 'D3', scale_factor=0.8)
        self.place_at_grid(chart_y, 'D5', scale_factor=0.8)
        
        self.play(FadeIn(chart_x), FadeIn(chart_y))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_Y))
        
        # Flip Chart Y (bars only to avoid flipping label text)
        bars_y = chart_y[0]
        self.play(bars_y.animate.scale([-1, 1, 1]))
        
        flipped_tag = Text("(Flipped)", font_size=16, color=COLOR_Y)
        flipped_tag.next_to(chart_y, DOWN, buff=0.1)
        self.play(Write(flipped_tag))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_HIGHLIGHT))
        
        # Slide Setup: Move Chart Y to the left of Chart X
        # Calculate grid-based positions to avoid manual positioning
        start_pos_y = self.grid['D1']
        formula_target_pos = (self.grid['A1'] + self.grid['A6']) / 2
        
        self.play(
            chart_y.animate.move_to(start_pos_y),
            flipped_tag.animate.next_to(start_pos_y, DOWN, buff=0.1),
            formula.animate.move_to(formula_target_pos).scale(0.8)
        )

        # Tracker for sliding animation
        slide_tracker = ValueTracker(0)
        
        # Attach updater for smooth sliding
        def update_y(m):
            m.move_to(start_pos_y + RIGHT * slide_tracker.get_value())
            
        chart_y.add_updater(update_y)
        # Separate updater for tag to keep it below chart_y
        flipped_tag.add_updater(lambda m: m.next_to(chart_y, DOWN, buff=0.1))
        
        # Slide through the chart X (from D1 to D5)
        # D1 is x=0.5, D5 is x=4.5. Distance is 4.0 units.
        self.play(slide_tracker.animate.set_value(4.0), run_time=5, rate_func=linear)
        
        # Pause at an overlap to show multiplication (back to D3 area)
        # D3 is x=2.5. Distance from D1 is 2.0.
        self.play(slide_tracker.animate.set_value(2.0), run_time=1)
        
        # Highlight overlap
        highlight_rect = Rectangle(width=0.45, height=1.8, color=COLOR_HIGHLIGHT, stroke_width=4)
        highlight_rect.move_to(chart_x[0][1].get_center())
        
        mult_expr = MathTex("0.5 \\times 0.4", color=COLOR_HIGHLIGHT, font_size=24)
        # Fix issue 24: position mult_expr at F5
        self.place_at_grid(mult_expr, 'F5', scale_factor=0.8)
        
        self.play(Create(highlight_rect), Write(mult_expr))
        self.wait(2)
        
        # Cleanup updaters
        chart_y.remove_updater(update_y)
        flipped_tag.clear_updaters()
