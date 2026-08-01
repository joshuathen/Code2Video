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
        title_str = "Conclusion and Key Takeaway"
        lines = [
            "Derivatives show how fast things change now.",
            "Integrals show how much has built up.",
            "Calculus bridges these two views of the world."
        ]
        self.setup_layout(title_str, lines)

        # === Animation for Lecture Line 1 ===
        # The text "Derivatives: How fast?" appears in green (#00FF00) with a small slope graphic.
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        deriv_text = Text("Derivatives: How fast?", font_size=24, color="#00FF00")
        self.place_in_area(deriv_text, "B2", "B3", scale_factor=0.8)
        
        # Slope graphic
        axes1 = Axes(
            x_range=[0, 3], 
            y_range=[0, 3], 
            axis_config={"include_tip": False},
            x_length=2,
            y_length=2
        )
        self.place_at_grid(axes1, "B5")
        curve1 = axes1.plot(lambda x: 0.2 * x**2 + 0.5, color=WHITE)
        dot1 = Dot(axes1.c2p(1.5, 0.2*1.5**2+0.5), color="#00FF00")
        slope_line = Line(axes1.c2p(0.5, 0.2), axes1.c2p(2.5, 1.7), color="#00FF00")
        slope_graphic = VGroup(axes1, curve1, dot1, slope_line)
        
        self.play(Write(deriv_text), Create(slope_graphic))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The text "Integrals: How much?" appears in blue (#1E90FF) with a small filled-area graphic.
        self.play(self.lecture[1].animate.set_color("#1E90FF"))
        
        int_text = Text("Integrals: How much?", font_size=24, color="#1E90FF")
        self.place_in_area(int_text, "D1", "D2", scale_factor=0.8)
        
        # Area graphic
        axes2 = Axes(
            x_range=[0, 3], 
            y_range=[0, 3], 
            axis_config={"include_tip": False},
            x_length=2,
            y_length=2
        )
        self.place_at_grid(axes2, "D5")
        curve2 = axes2.plot(lambda x: 0.2 * x**2 + 0.5, color=WHITE)
        area = axes2.get_area(curve2, x_range=[0.5, 2.5], color="#1E90FF", opacity=0.5)
        area_graphic = VGroup(axes2, curve2, area)
        
        self.play(Write(int_text), Create(area_graphic))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Both lines of text converge and transform into a bright "Calculus Unified" title.
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        unified_text = Text("Calculus Unified", font_size=36, color="#FFD700")
        self.place_in_area(unified_text, "E4", "F5", scale_factor=1.0)
        
        self.play(
            FadeOut(slope_graphic),
            FadeOut(area_graphic),
            Transform(deriv_text, unified_text),
            Transform(int_text, unified_text),
            run_time=2
        )
        self.remove(int_text) # Clean up duplicate after convergence
        self.wait(3)
