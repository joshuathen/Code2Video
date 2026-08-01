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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Mechanics: Flip, Shift, and Multiply"
        lecture_lines = [
            "Convolution follows a simple three-step physical process.",
            "First, we flip the filter kernel horizontally.",
            "Next, we shift the flipped filter across the input.",
            "We multiply overlapping values and sum the results.",
            "This sum becomes the output at that specific position."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_X = BLUE_C
        COLOR_H = YELLOW_C
        COLOR_OVERLAP = "#00FF00"
        COLOR_Y = ORANGE

        # === Animation for Lecture Line 1 ===
        # Convolution follows a simple three-step physical process.
        self.lecture[0].set_color(COLOR_X)
        
        # Input Signal x[k] at Row C
        x_values = [1.0, 1.5, 2.0, 1.5, 1.0]
        x_bars = VGroup(*[
            Rectangle(width=0.6, height=v*0.5, fill_opacity=0.8, fill_color=COLOR_X, stroke_width=1)
            for v in x_values
        ])
        # Position x_bars at C2-C6
        for i, bar in enumerate(x_bars):
            self.place_at_grid(bar, f"C{i+2}")
        
        x_label = MathTex("x[k]", color=COLOR_X, font_size=24)
        self.place_at_grid(x_label, "C1")
        
        self.play(Create(x_bars), Write(x_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # First, we flip the filter kernel horizontally.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_H)
        
        # Kernel h[k] - starting at B4, B5, B6 (Issue 32 fix)
        h_vals = [0.4, 0.8, 1.2]
        h_bars = VGroup(*[
            Rectangle(width=0.6, height=v*0.5, fill_opacity=0.8, fill_color=COLOR_H, stroke_width=1)
            for v in h_vals
        ])
        for i, bar in enumerate(h_bars):
            self.place_at_grid(bar, f"B{i+4}")
            
        h_label = MathTex("h[k]", color=COLOR_H, font_size=24)
        self.place_at_grid(h_label, "B3") # Within 1 unit of B4
        
        self.play(Create(h_bars), Write(h_label))
        self.wait(0.5)
        
        # Flip animation: Swap positions of bars relative to B5
        flip_label = MathTex("h[-k]", color=COLOR_H, font_size=24)
        self.place_at_grid(flip_label, "B2") # Issue 33 fix

        self.play(
            h_bars[0].animate.move_to(self.grid["B6"]),
            h_bars[2].animate.move_to(self.grid["B4"]),
            Transform(h_label, flip_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Next, we shift the flipped filter across the input.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Re-order and group for sliding (h_bars[2] is now leftmost)
        flipped_kernel = VGroup(h_bars[2], h_bars[1], h_bars[0])
        
        # Move down towards row C (positioned slightly above to avoid overlap with x_bars)
        self.play(flipped_kernel.animate.shift(DOWN * 0.8), run_time=1)
        
        # Slide across from left to right
        for i in range(1, 5):
            self.play(flipped_kernel.animate.move_to(self.grid[f"C{i}"] + UP*0.3), run_time=0.8)
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We multiply overlapping values and sum the results.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_OVERLAP)
        
        # Currently flipped_kernel is centered at C4. Bars at C3, C4, C5. 
        # These overlap with x_bars (which are at C2, C3, C4, C5, C6).
        overlap_indices = [3, 4, 5]
        highlights = VGroup(*[
            Rectangle(width=0.7, height=1.3, color=COLOR_OVERLAP, stroke_width=2).move_to(self.grid[f"C{i}"])
            for i in overlap_indices
        ])
        
        multipliers = VGroup(*[
            MathTex("\\times", color=COLOR_OVERLAP, font_size=20).move_to(self.grid[f"C{i}"] + UP*0.1)
            for i in overlap_indices
        ])
        
        sigma = MathTex("\\Sigma", color=COLOR_OVERLAP, font_size=36)
        self.place_at_grid(sigma, "D4")
        
        self.play(Create(highlights), FadeIn(multipliers))
        self.play(Write(sigma))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This sum becomes the output at that specific position.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_Y)
        
        # Output axis at row E
        y_axis = Line(self.grid["E1"], self.grid["E6"], color=WHITE)
        y_label = MathTex("y[n]", color=COLOR_Y, font_size=24)
        self.place_at_grid(y_label, "E3") # Issue 34 fix: Closer to data path
        
        # The result dot at position corresponding to C4 (center of kernel)
        result_dot = Dot(color=COLOR_Y).move_to(self.grid["E4"])
        
        self.play(Create(y_axis), Write(y_label))
        
        # Visual transition of the calculation into a point on the output signal
        self.play(
            Succession(
                FadeOut(multipliers, highlights, shift=DOWN),
                Transform(sigma, result_dot)
            ),
            run_time=2
        )
        self.wait(2)
