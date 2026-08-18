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
            "Consider an accumulation function, A(x).",
            "It represents area under f(t).",
            "Increasing x adds a tiny rectangular slice.",
            "The slice height is f(x).",
            "Therefore, A'(x) = f(x)."
        ]
        self.setup_layout("Visualizing the Fundamental Theorem", lecture_lines)
        
        # Setup Axes and Curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda t: 0.5 * t**2 + 0.5, x_range=[0, 3.5])
        
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, 'C1', 'F6', scale_factor=0.5)
        
        # Load Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        pen = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pen.svg")
        
        self.place_at_grid(ruler, 'B3', scale_factor=0.3)
        self.place_at_grid(pen, 'C3', scale_factor=0.3)
        
        a = 0.5
        x_tracker = ValueTracker(2.0)
        
        # Area under curve - using persistent mobject (set opacity via set_fill)
        area = Polygon(axes.c2p(a, 0), axes.c2p(2.0, 0), axes.c2p(2.0, 0.5 * 2.0**2 + 0.5), axes.c2p(a, 0.5 * a**2 + 0.5), color=GREEN).set_fill(opacity=0.4)
        
        def update_area(m):
            x_val = x_tracker.get_value()
            new_area = axes.get_area(curve, x_range=[a, x_val], color=GREEN).set_fill(opacity=0.4)
            m.become(new_area)
            
        area.add_updater(update_area)
        
        self.add(area, graph_group, ruler, pen)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(x_tracker.animate.set_value(3.0), pen.animate.move_to(axes.c2p(3.0, 0.5 * 3.0**2 + 0.5)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.wait(2)
