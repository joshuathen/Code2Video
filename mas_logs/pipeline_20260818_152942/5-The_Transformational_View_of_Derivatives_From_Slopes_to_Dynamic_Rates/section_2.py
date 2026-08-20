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
        self.setup_layout("The Transformation: The Limit Process", [
            "Let the gap between two points shrink to zero.",
            "As they converge, the secant becomes the tangent.",
            "The line snaps into the instantaneous position.",
            "This limit defines the derivative geometrically.",
            "It transforms interval change into a single point."
        ])
        
        # Setup plot area
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.8)
        
        func = lambda x: 0.2 * x**2 + 1
        curve = axes.plot(func, color=WHITE)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Points
        p1 = axes.c2p(2, func(2))
        dot1 = Dot(p1, color=YELLOW)
        dot2 = Dot(axes.c2p(4, func(4)), color=YELLOW)
        
        # Secant Line
        secant = Line(p1, axes.c2p(4, func(4)), color="#FFD700")
        
        graph_group = VGroup(axes, curve, dot1, dot2, secant)
        self.place_in_area(graph_group, 'B4', 'F6', scale_factor=0.75)
        self.add(graph_group)
        
        h = ValueTracker(2.0)
        
        def update_secant(mob):
            new_p2 = axes.c2p(2 + h.get_value(), func(2 + h.get_value()))
            mob.put_start_and_end_on(p1, new_p2)
            dot2.move_to(new_p2)
            
        secant.add_updater(update_secant)
        
        limit_text = MathTex(r"\lim_{h \to 0}", font_size=36, color="#00BFFF")
        self.place_at_grid(limit_text, 'C4')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_at_grid(ruler, 'B2')
        self.play(FadeIn(ruler))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF4500")
        self.play(h.animate.set_value(0.1), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        secant.remove_updater(update_secant)
        secant.set_color(WHITE)
        self.play(FadeOut(dot2), FadeOut(ruler), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00BFFF")
        self.play(Write(limit_text))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#32CD32")
        self.place_at_grid(protractor, 'D2')
        self.play(FadeIn(protractor))
        self.wait(2)
