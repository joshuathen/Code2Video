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
            "Energy follows -5/3 power law.",
            "Plot log energy versus wave number.",
            "Slope -5/3 marks inertial range.",
            "Eddies are self-similar fractal structures.",
            "Geometry repeats across scale ranges."
        ]
        self.setup_layout("The Mathematical Structure: The -5/3 Power Law", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # B038: Ensure sqrt or specific terms are clear if needed.
        # B016: Row A/top of Row B kept clear.
        eq = MathTex(r"E(k) \propto k^{-5/3}", color="#FFFF00")
        self.place_at_grid(eq, 'B3', scale_factor=1.0)
        self.play(Write(eq))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # B048: Constrain scale.
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        plot = axes.plot(lambda x: 3 - x, x_range=[0.5, 2.5], color="#FFFFFF")
        graph = VGroup(axes, plot)
        self.place_in_area(graph, 'D3', 'F6', scale_factor=0.5)
        self.play(Create(axes), Create(plot))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # B045: Restrict scale of secondary emphasis.
        slope_label = MathTex(r"\text{slope} = -5/3", color="#FF0000").scale(0.8)
        self.place_at_grid(slope_label, 'C5')
        self.play(Write(slope_label))
        self.lecture[2].set_color("#FF0000")

        # === Animation for Lecture Line 4 ===
        circle = Circle(radius=0.5, color="#00FFFF")
        self.place_at_grid(circle, 'C4', scale_factor=0.4)
        self.play(GrowFromCenter(circle))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        circle2 = Circle(radius=0.25, color="#00FF00")
        self.place_at_grid(circle2, 'C5', scale_factor=0.4)
        self.play(GrowFromCenter(circle2))
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
