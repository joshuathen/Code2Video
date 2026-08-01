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
        self.setup_layout(
            "Prerequisite: The Multi-Dimensional Pythagorean Theorem", 
            [
                "Pythagorean theorem extends to any number of dimensions.",
                "We sum the squares of all coordinate values.",
                "This defines the distance to origin in n-dimensions."
            ]
        )
        
        # Colors for matching lecture lines
        COLOR_1 = "#00FF00" # Green
        COLOR_2 = "#FFFF00" # Yellow
        COLOR_3 = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        # Display the 2D formula x^2 + y^2 = R^2 in green (#00FF00)
        self.lecture[0].set_color(COLOR_1)
        formula_2d = MathTex("x^2 + y^2 = R^2", color=COLOR_1)
        # Resolved Issue 25: scale_factor changed from 1.5 to 1.1
        self.place_in_area(formula_2d, "C3", "D4", scale_factor=1.1)
        
        self.play(Write(formula_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the 2D formula morphing into n-dimensional formula in yellow (#FFFF00).
        self.lecture[1].set_color(COLOR_2)
        formula_nd = MathTex("x_1^2 + x_2^2 + \\dots + x_n^2 = R^2", color=COLOR_2)
        # Resolved Issue 26: Changed area to "C3"-"E6" and scale_factor to 1.0 to avoid lecture text overlap
        self.place_in_area(formula_nd, "C3", "E6", scale_factor=1.0)
        
        self.play(Transform(formula_2d, formula_nd))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply a pulsing white (#FFFFFF) highlight around the entire n-dimensional sum.
        self.lecture[2].set_color(COLOR_3)
        
        # Use a Rectangle that updates its size and position based on the formula
        highlight_rect = SurroundingRectangle(formula_2d, color=COLOR_3, buff=0.2)
        self.play(Create(highlight_rect))
        
        # Pulse animation using ValueTracker for efficiency
        pulse_tracker = ValueTracker(1.0)
        
        def pulse_rect_update(m):
            val = pulse_tracker.get_value()
            m.set_width(formula_2d.width * val + 0.4)
            m.set_height(formula_2d.height * val + 0.4)
            m.move_to(formula_2d.get_center())

        highlight_rect.add_updater(pulse_rect_update)
        
        self.play(pulse_tracker.animate.set_value(1.15), run_time=0.8, rate_func=there_and_back)
        self.play(pulse_tracker.animate.set_value(1.15), run_time=0.8, rate_func=there_and_back)
        
        self.wait(2)
        
        # Clean up
        highlight_rect.clear_updaters()
        self.play(FadeOut(highlight_rect), FadeOut(formula_2d))
        self.wait(1)
