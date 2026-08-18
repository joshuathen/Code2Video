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
            "Tangent lines approximate curves linearly.",
            "Zoom in to see straight lines.",
            "Curves become simple linear maps.",
            "The slope is the scale.",
            "Linearity simplifies complex motion."
        ]
        self.setup_layout("Visualizing the Linear Approximation", lecture_lines)

        # Assets
        rubber = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rubber.svg")
        sheet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")

        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        curve = FunctionGraph(lambda x: 0.2 * x**3 + 0.1 * x**2 - 0.5 * x, color="#FFFFFF")
        graph_group = VGroup(axes, curve, rubber, sheet)
        
        self.place_in_area(graph_group, 'C3', 'F6', scale_factor=0.6)
        self.place_at_grid(rubber, 'A4', scale_factor=0.7)
        self.place_at_grid(sheet, 'A5', scale_factor=0.7)
        
        self.play(Create(axes), Create(curve), FadeIn(rubber), FadeIn(sheet))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        zoom_box = Rectangle(color="#3498DB", height=1.5, width=1.5)
        self.place_in_area(zoom_box, 'C3', 'E5', scale_factor=0.5)
        self.add(zoom_box)
        
        self.play(zoom_box.animate.scale(2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#3498DB")
        tangent = Line(start=LEFT*1, end=RIGHT*1, color="#E74C3C")
        tangent.move_to(curve.point_from_proportion(0.5))
        self.play(FadeIn(tangent))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#E74C3C")
        self.play(Indicate(tangent))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#2ECC71")
        self.play(FadeOut(zoom_box), FadeOut(graph_group), FadeOut(tangent))
        self.place_at_grid(sheet, 'C3', scale_factor=1.5)
        self.play(FadeIn(sheet))
        self.wait(2)
