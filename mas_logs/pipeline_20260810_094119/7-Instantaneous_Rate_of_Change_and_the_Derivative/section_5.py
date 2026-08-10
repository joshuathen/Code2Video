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
        self.setup_layout("Summary and Synthesis", [
            "Derivative equals the tangent line's slope.",
            "Average change versus instantaneous focus.",
            "Derivative provides precision for engineering."
        ])
        
        # Visuals
        summary_graph = VGroup(
            Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False}).scale(0.5),
            FunctionGraph(lambda x: x**2 / 4, x_range=[0, 4]).set_color(BLUE)
        )
        
        derivative_formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}").scale(0.8)
        takeaway = Text("Derivative = Precision", font_size=32)
        
        # Assets: Ensure these paths exist and are valid SVG files. 
        # PNG files are not compatible with SVGMobject; use ImageMobject for PNGs.
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        speedometer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(summary_graph, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(summary_graph))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(derivative_formula, 'E3', scale_factor=0.7)
        self.play(Write(derivative_formula))
        self.lecture[1].set_color("#FF4500")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(summary_graph), FadeOut(derivative_formula))
        self.place_at_grid(takeaway, 'E4', scale_factor=0.9)
        self.play(FadeIn(takeaway))
        
        # Overlay assets
        self.place_at_grid(car_icon, 'C2', scale_factor=0.5)
        self.place_at_grid(speedometer_icon, 'C4', scale_factor=0.5)
        speedometer_icon.set_color("#00FF00")
        
        self.play(FadeIn(car_icon), FadeIn(speedometer_icon))
        
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
