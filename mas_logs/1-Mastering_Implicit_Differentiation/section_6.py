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

class Section6Scene(TeachingScene):
    def construct(self):
        # Mandatory layout setup
        self.setup_layout("Summary & Key Takeaways", [
            "Remember: always tag y derivatives with dy dx.",
            "Isolate dy dx to find the rate of change.",
            "You can now differentiate any relation, no matter how messy."
        ])

        # === Animation for Lecture Line 1 ===
        # Animation Description: Display 'Whenever differentiating y, multiply by dy/dx' in Gold (#FFD700).
        golden_rule = Text("Whenever differentiating y,\nmultiply by dy/dx", color="#FFD700", font_size=28)
        self.place_in_area(golden_rule, "A1", "B6")
        
        self.play(
            self.lecture[0].animate.set_color("#FFD700"),
            Write(golden_rule)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animation Description: Checklist appears: 'Chain Rule?', 'Isolate dy/dx?', 'Slope found?' with green checks (#00FF00).
        
        # Define checklist text
        item1 = Text("Chain Rule?", font_size=24)
        item2 = Text("Isolate dy/dx?", font_size=24)
        item3 = Text("Slope found?", font_size=24)
        
        # Define checkmarks
        check1 = Text("✓", color="#00FF00", font_size=32)
        check2 = Text("✓", color="#00FF00", font_size=32)
        check3 = Text("✓", color="#00FF00", font_size=32)

        # Position checklist items on the grid
        # Fixed: Using place_in_area for multi-word labels to improve visual flow and avoid cramping (Issue 25, 26)
        self.place_in_area(item1, 'C2', 'C3', scale_factor=0.8)
        self.place_at_grid(check1, "C4")
        
        self.place_in_area(item2, 'D2', 'D3', scale_factor=0.8)
        self.place_at_grid(check2, "D4")
        
        self.place_in_area(item3, 'E2', 'E3', scale_factor=0.8)
        self.place_at_grid(check3, "E4")

        # Color the lecture line and reveal checklist
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        self.play(FadeIn(item1), FadeIn(check1))
        self.play(FadeIn(item2), FadeIn(check2))
        self.play(FadeIn(item3), FadeIn(check3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animation Description: The 'dy/dx' symbol pulses in bright green (#00FF00) for emphasis.
        dydx_symbol = Text("dy/dx", color="#00FF00", font_size=80)
        # Fixed: Adjusted area to F2-F5 for better balance (Issue 27)
        self.place_in_area(dydx_symbol, 'F2', 'F5', scale_factor=1.0)
        
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(dydx_symbol))
        
        # Pulse effect: scale up and down twice
        self.play(dydx_symbol.animate.scale(1.25), run_time=0.3)
        self.play(dydx_symbol.animate.scale(1/1.25), run_time=0.3)
        self.play(dydx_symbol.animate.scale(1.25), run_time=0.3)
        self.play(dydx_symbol.animate.scale(1/1.25), run_time=0.3)
        
        self.wait(3)
