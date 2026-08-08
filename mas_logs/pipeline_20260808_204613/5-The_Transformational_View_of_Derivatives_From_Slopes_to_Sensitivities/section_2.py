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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Consider a simple curve.", 
            "Draw a secant line between two points.", 
            "Slide the second point toward the first.", 
            "The secant converges to a tangent line.", 
            "This line defines our instantaneous slope."
        ]
        self.setup_layout("Prerequisite Visual: The Secant-to-Tangent Transition", lecture_lines)
        
        # Define curve
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_numbers": False}).scale(0.5)
        curve = axes.plot(lambda x: x**2 + 0.5, color=WHITE)
        
        # Applying instruction 26 to avoid obstruction
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'B4', 'F6', scale_factor=0.75)
        
        # Points
        x1, x2 = -1, 1
        p1 = axes.c2p(x1, x1**2 + 0.5)
        p2 = axes.c2p(x2, x2**2 + 0.5)
        
        dot1 = Dot(p1, color=BLUE)
        dot2 = Dot(p2, color=RED)
        
        # Secant
        secant = Line(p1, p2, color="#FFFFFF")
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Using a fallback for the provided asset as it might not be a standard image. 
        # Since the path exists in the prompt, let's treat it as an SVG icon
        # Try loading, if fails, use a fallback shape.
        try:
            icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        except:
            icon = Dot(color=WHITE)
        self.place_at_grid(icon, 'A2', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(graph))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(Create(dot1), Create(dot2), Create(secant))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        # Use ValueTracker for smooth slide
        t = ValueTracker(x2)
        secant.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(x1, x1**2 + 0.5), 
            axes.c2p(t.get_value(), t.get_value()**2 + 0.5)
        ))
        dot2.add_updater(lambda m: m.move_to(axes.c2p(t.get_value(), t.get_value()**2 + 0.5)))
        
        self.play(t.animate.set_value(x1 + 0.1), run_time=2)
        self.lecture[2].set_color("#FF5733")

        # === Animation for Lecture Line 4 ===
        secant.set_color("#33FF57")
        self.lecture[3].set_color("#33FF57")
        self.play(FadeIn(icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F3FF33")
        self.wait(2)
