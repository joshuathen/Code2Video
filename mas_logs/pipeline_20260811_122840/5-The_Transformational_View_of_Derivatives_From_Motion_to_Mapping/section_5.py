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
        self.setup_layout("Derivative: Local Geometry Modifier", [
            "Derivative is a local modifier.",
            "It warps the input space.",
            "Compare constant, linear, quadratic stretches."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the core idea: Derivative = Local Geometry Modifier.
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg")
        title_idea = Text("Derivative = Local Geometry Modifier", font_size=32, color="#FFFFFF")
        group_1 = VGroup(title_idea, microscope).arrange(RIGHT, buff=0.2)
        self.place_at_grid(group_1, "B5", scale_factor=0.6)
        self.play(Write(title_idea), FadeIn(microscope))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the zoom-in from global curve to local tangent.
        curve = FunctionGraph(lambda x: 0.1 * x**3 - 0.5 * x, x_range=[-3, 3], color=GRAY)
        self.place_in_area(curve, 'C4', 'E6', scale_factor=0.6)
        self.play(Create(curve))
        
        magnifyingglass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        zoom_circle = Circle(radius=0.5, color="#3498DB")
        zoom_group = VGroup(zoom_circle, magnifyingglass)
        self.place_at_grid(zoom_group, 'D5', scale_factor=0.7)
        self.play(Create(zoom_circle), FadeIn(magnifyingglass))
        self.lecture[1].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the transition from space to flat approximation.
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        flat_line = Line(start=LEFT, end=RIGHT, color="#E74C3C")
        self.place_at_grid(flat_line, 'F2', scale_factor=0.7)
        
        comparison = VGroup(
            Text("Space Warping:", font_size=24),
            flat_line,
            ruler
        ).arrange(DOWN)
        self.place_at_grid(comparison, 'F5', scale_factor=0.7)
        
        self.play(FadeIn(comparison))
        self.lecture[2].set_color("#E74C3C")
        self.wait(2)
