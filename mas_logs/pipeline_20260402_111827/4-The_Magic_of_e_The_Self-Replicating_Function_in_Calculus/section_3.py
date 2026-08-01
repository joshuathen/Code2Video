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
        # Setup base scene
        lecture_lines = [
            "Let's compare different exponential growth rates.",
            "At zero, the slope of 2^x is 0.69.",
            "The slope of 3^x is steeper, at 1.10.",
            "We want a slope that perfectly matches the height.",
            "We need a base between two and three."
        ]
        self.setup_layout("The Search for the Perfect Base", lecture_lines)

        # Define Colors
        COLOR_2X = "#FF0000"  # Red
        COLOR_3X = "#0000FF"  # Blue
        COLOR_TARGET = "#00FF00"  # Green

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create axes on the right side
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4.5,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, 'A1', 'F6')
        self.play(Create(axes))
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        curve_2x = axes.plot(lambda x: 2**x, x_range=[-2, 1.8], color=COLOR_2X)
        # Calculate alpha for x=0 in the range [-2, 1.8]: (0 - (-2)) / (1.8 - (-2)) = 2 / 3.8
        tangent_2x = TangentLine(curve_2x, alpha=2/3.8, length=4).set_color(COLOR_2X)
        slope_label_2x = Text("m ≈ 0.69", font_size=20, color=COLOR_2X)
        # Resolved Issue 44: Positioning red slope label at D6
        self.place_at_grid(slope_label_2x, 'D6', scale_factor=0.8)
        
        self.play(Create(curve_2x))
        self.play(Create(tangent_2x), Write(slope_label_2x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        curve_3x = axes.plot(lambda x: 3**x, x_range=[-2, 1.2], color=COLOR_3X)
        # Calculate alpha for x=0 in the range [-2, 1.2]: (0 - (-2)) / (1.2 - (-2)) = 2 / 3.2
        tangent_3x = TangentLine(curve_3x, alpha=2/3.2, length=4).set_color(COLOR_3X)
        slope_label_3x = Text("m ≈ 1.10", font_size=20, color=COLOR_3X)
        # Resolved Issue 43: Positioning blue slope label at B3
        self.place_at_grid(slope_label_3x, 'B3', scale_factor=0.8)
        
        self.play(Create(curve_3x))
        self.play(Create(tangent_3x), Write(slope_label_3x))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        # Highlight the point (0,1) where height is 1
        dot = Dot(axes.c2p(0, 1), color=WHITE)
        height_label = Text("y = 1", font_size=20)
        # Resolved Issue 45: Positioning y-intercept label at E5
        self.place_at_grid(height_label, 'E5', scale_factor=0.7)
        
        # Target line (slope 1 at x=0): y = x + 1
        target_line = DashedVMobject(axes.plot(lambda x: x + 1, x_range=[-1, 1], color=COLOR_TARGET))
        
        self.play(FadeIn(dot), Write(height_label))
        self.play(Create(target_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Shade the region between the two curves
        shading = axes.get_area(curve_3x, x_range=[-2, 1.2], bounded_graph=curve_2x, color=WHITE, opacity=0.2)
        
        # Suggest e^x
        curve_ex = axes.plot(lambda x: np.exp(x), x_range=[-2, 1.3], color=COLOR_TARGET)
        label_ex = Text("e^x", font_size=24, color=COLOR_TARGET)
        self.place_at_grid(label_ex, 'A6', scale_factor=1.0)
        
        self.play(FadeIn(shading))
        self.play(Create(curve_ex), Write(label_ex))
        self.wait(2)
