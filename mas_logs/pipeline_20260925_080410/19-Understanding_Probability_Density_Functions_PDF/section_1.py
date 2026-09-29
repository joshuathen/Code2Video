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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: From Discrete to Continuous", 
                          ["Discrete probability uses bars for distinct outcomes.", 
                           "Continuous variables mean zero probability at exact points.", 
                           "Focus shifts from bar height to area under curves."])
        
        # === Animation for Lecture Line 1 ===
        # Show text 'Discrete Probability' at grid B2
        discrete_text = Text("Discrete Probability", color="#FF5733", font_size=20)
        self.place_at_grid(discrete_text, 'B2')
        
        bar_chart = VGroup(*[Rectangle(height=0.5 + i*0.2, width=0.4, color=WHITE).set_fill(WHITE, opacity=0.8) for i in range(4)])
        bar_chart.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        self.place_at_grid(bar_chart, 'C2', scale_factor=0.8)
        
        discrete_group = VGroup(discrete_text, bar_chart)
        self.play(Write(discrete_text), Create(bar_chart), self.lecture[0].animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show text 'Continuous Probability' at grid B5
        continuous_text = Text("Continuous Probability", color="#33FF57", font_size=20)
        self.place_at_grid(continuous_text, 'B5')
        
        axes = Axes(x_range=[-2, 2], y_range=[0, 1], axis_config={"include_numbers": False}).scale(0.3)
        curve = axes.plot(lambda x: np.exp(-x**2), x_range=[-2, 2], color=WHITE)
        self.place_at_grid(curve, 'C5', scale_factor=0.8)
        
        self.play(Write(continuous_text), Create(curve), self.lecture[1].animate.set_color("#33FF57"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight area under the curve
        area = axes.get_area(curve, x_range=[-2, 2], color="#FFFFFF", opacity=0.5)
        self.place_at_grid(area, 'C5', scale_factor=0.8)
        
        self.play(FadeIn(area), self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(2)
