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
        # 1. Setup Layout
        title_text = "The Mathematical Logic: Mean and Spread"
        lecture_lines = [
            "The bell curve centers exactly at the population mean.",
            "Increasing sample size makes the bell curve much narrower.",
            "Larger samples provide a more precise estimate of truth.",
            "This relationship follows the predictable Standard Error formula.",
            "Precision grows as the square root of sample size."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Apply color changes to lecture lines for visual association
        line_colors = [WHITE, "#FF00FF", "#00FFFF", WHITE, WHITE]
        for i, color in enumerate(line_colors):
            self.lecture[i].set_color(color)

        # Helper function for bell curves (Normal Distribution)
        # Using a fixed amplitude or scaling to fit the grid nicely
        def get_bell_curve(sigma, color):
            # Normal distribution pdf: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * (x/sigma)^2)
            # We scale the x-axis to fit our grid better.
            return FunctionGraph(
                lambda x: 1.5 * np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi)) * (sigma/1.0),
                x_range=[-2.5, 2.5],
                color=color
            )

        # === Animation for Lecture Line 1 ===
        # "The bell curve centers exactly at the population mean."
        # Use a vertical dashed line and label mu.
        mu_line = DashedLine(start=DOWN * 1.5, end=UP * 1.5, color=WHITE, stroke_width=2)
        mu_label = MathTex(r"\mu", color=WHITE)
        
        # Position mean line at D4 per Issue 34
        self.place_at_grid(mu_line, "D4")
        mu_label.next_to(mu_line, DOWN, buff=0.1)
        
        # Background wide curve (initially purple)
        curve_wide = get_bell_curve(1.2, "#FF00FF")
        self.place_at_grid(curve_wide, "D4") # Center at D4 per Issue 34

        self.play(Create(curve_wide))
        self.play(Create(mu_line), Write(mu_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Increasing sample size makes the bell curve much narrower."
        label_small = Text("n=5", font_size=20, color="#FF00FF")
        # Position label_small per Issue 33
        self.place_in_area(label_small, 'C2', 'C3')
        
        self.play(Write(label_small))
        self.wait(1)

        # Transitioning to a narrower distribution (Large n)
        curve_narrow = get_bell_curve(0.5, "#00FFFF")
        self.place_at_grid(curve_narrow, "D4") # Center at D4 per Issue 34
        
        label_large = Text("n=100", font_size=20, color="#00FFFF")
        # Position label_large per Issue 33
        self.place_in_area(label_large, 'C5', 'C6')
        
        self.play(
            Transform(curve_wide, curve_narrow),
            Write(label_large),
            label_small.animate.set_fill(opacity=0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Larger samples provide a more precise estimate of truth."
        # Flash the peak of the narrow cyan curve to indicate high precision.
        peak_point = self.grid["D4"] + UP * 1.2 # Approximate peak location
        flash_peak = Dot(peak_point, radius=0.01, color="#00FFFF", fill_opacity=0)
        
        self.play(Flash(flash_peak, color="#00FFFF", line_length=0.3, num_lines=8))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This relationship follows the predictable Standard Error formula."
        # FadeIn formula at top.
        formula = MathTex(
            r"\text{Standard Error} = \frac{\sigma}{\sqrt{n}}",
            color=WHITE
        )
        # Position formula per Issue 35
        self.place_in_area(formula, 'A3', 'A5', scale_factor=0.8)
        
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Precision grows as the square root of sample size."
        # Highlight the \sqrt{n} part. 
        # In MathTex, "Standard Error = \frac{\sigma}{\sqrt{n}}", 
        # parts are roughly divided by symbols if using multiple strings, 
        # but here we use one. We use get_part_by_tex or indexing.
        # \sqrt{n} is the last part.
        
        sqrt_n = formula[0][-3:] # \sqrt{n} is the last 3 characters roughly in rendering terms
        # More robust way:
        formula_parts = MathTex(
            r"\text{Standard Error} =", r"\frac{\sigma}{\sqrt{n}}",
            color=WHITE
        )
        self.place_in_area(formula_parts, 'A3', 'A5', scale_factor=0.8)
        self.remove(formula)
        self.add(formula_parts)
        
        # Narrow the curve further.
        curve_extra_narrow = get_bell_curve(0.25, "#00FFFF")
        self.place_at_grid(curve_extra_narrow, "D4")
        
        # Highlight \sqrt{n}
        # In \frac{\sigma}{\sqrt{n}}, indices: 0: \sigma, 1: bar, 2: \sqrt, 3: n
        # This depends on Manim version. Let's use a simpler approach: SurroundingRectangle
        highlight_box = SurroundingRectangle(formula_parts[1][2:], color=YELLOW, buff=0.05)

        self.play(
            Create(highlight_box),
            Transform(curve_wide, curve_extra_narrow),
            run_time=2
        )
        self.play(FadeOut(highlight_box))
        self.wait(2)
