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
        # Setup the layout with Title and Lecture Lines
        title = "The Time Function"
        lines = [
            "Total time is distance divided by velocity.",
            "Sum the travel times for both media.",
            "We have a time function depending on x."
        ]
        self.setup_layout(title, lines)

        # Colors for consistency
        V1_COLOR = "#00FF00"
        V2_COLOR = "#00FFFF"
        TEXT_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Principle: T = d / v
        eq_principle = Text("T = d / v", font_size=40, color=TEXT_WHITE)
        # Fix Issue 38: Move to A3 for visual hierarchy
        self.place_at_grid(eq_principle, "A3", scale_factor=1.0)
        
        self.play(FadeIn(eq_principle))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Display the total time equation T(x) = T1 + T2 in the top center
        eq_sum = Text("T(x) = T₁ + T₂", font_size=38, color=TEXT_WHITE)
        self.place_at_grid(eq_sum, "B3", scale_factor=1.0)
        
        # Expand the equation to T(x) = d1/v1 + d2/v2
        eq_expand = MarkupText(
            f"T(x) = d₁/<span color='{V1_COLOR}'>v₁</span> + d₂/<span color='{V2_COLOR}'>v₂</span>",
            font_size=38
        )
        self.place_at_grid(eq_expand, "C3", scale_factor=1.0)
        
        self.play(ReplacementTransform(eq_principle, eq_sum))
        self.wait(1)
        self.play(FadeIn(eq_expand, shift=DOWN))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Substitute the square root expressions for d1 and d2 into the T(x) formula
        # T(x) = sqrt(x² + a²)/v1 + sqrt((w-x)² + b²)/v2
        # Using Unicode characters: √, ₁, ₂, ²
        final_func_str = (
            f"T(x) = √(x² + a²)/<span color='{V1_COLOR}'>v₁</span> + "
            f"√((w - x)² + b²)/<span color='{V2_COLOR}'>v₂</span>"
        )
        eq_final = MarkupText(final_func_str, font_size=30)
        
        # Fix Issue 39 & 40: Corrected area and scale factor
        self.place_in_area(eq_final, "D1", "D6", scale_factor=0.8)
        
        # Group previous terms to transform them into the final form
        self.play(
            ReplacementTransform(VGroup(eq_sum, eq_expand), eq_final),
            run_time=2
        )
        self.wait(2)
        
        # Reset color at the end
        self.lecture[2].set_color(WHITE)
        self.wait(1)
