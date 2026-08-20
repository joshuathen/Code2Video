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
        self.setup_layout("Visualizing the Limit: From Secant to Tangent", [
            "We plot position versus time on a graph. [Asset: curve_plot]",
            "A secant line connects two distinct points. [Asset: secant_line]",
            "Shrinking the interval makes the secant move. [Asset: shrinking_interval]",
            "As interval nears zero, it becomes tangent. [Asset: tangent_line]",
            "This limit gives us the instantaneous rate. [Asset: limit_arrow]"
        ])
        self.lecture.set_opacity(0)
        
        # Load asset
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"include_tip": True}).scale(0.5)
        curve = axes.plot(lambda x: 0.2 * x**3, color=WHITE)
        
        # Requirement: Move curve_plot to 'B3'-'F6'
        self.place_in_area(VGroup(axes, curve, graph_icon), "B3", "F6", scale_factor=0.85)
        
        t1 = ValueTracker(1)
        t2 = ValueTracker(3)
        
        def get_secant():
            p1 = axes.c2p(t1.get_value(), 0.2 * t1.get_value()**3)
            p2 = axes.c2p(t2.get_value(), 0.2 * t2.get_value()**3)
            return Line(p1, p2, color="#FF5733")
            
        secant = always_redraw(get_secant)
        
        # Requirement: Move secant_line to 'C3'-'E5'
        self.place_in_area(secant, "C3", "E5", scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(graph_icon), Create(axes), Create(curve))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733").set_opacity(1)
        self.add(secant)
        self.play(Indicate(secant))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57").set_opacity(1)
        self.play(t2.animate.set_value(1.1), run_time=2)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#33FF57").set_opacity(1)
        tangent = axes.plot(lambda x: 0.6 * x - 0.4, color="#33FF57")
        # Requirement: Move tangent_line to 'B3'-'E6'
        self.place_in_area(tangent, "B3", "E6", scale_factor=0.8)
        self.play(FadeIn(tangent), FadeOut(secant))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW).set_opacity(1)
        limit_arrow = Arrow(start=UP, end=ORIGIN, color="#33FF57")
        limit_arrow.next_to(tangent, UP)
        self.play(GrowArrow(limit_arrow))
        self.wait(1)
