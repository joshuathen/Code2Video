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
        self.setup_layout("The Concept of 'Zooming In'", [
            "What if the two points get closer?",
            "The secant line tilts as we zoom in.",
            "It eventually looks like a single point."
        ])
        
        # Create function graph
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.25 * x**2 + 1, x_range=[0, 4])
        graph_group = VGroup(axes, curve)
        
        # Asset: camera icon
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        
        # Using VideoCritic constraints:
        # 1. graph_group: place_in_area(graph_group, 'B4', 'E6', scale_factor=0.6)
        self.place_in_area(graph_group, 'B4', 'E6', scale_factor=0.6)
        self.place_in_area(camera_icon, 'A1', 'B2', scale_factor=0.3)
        
        # Points and lines
        p1 = Dot(color=BLUE).move_to(axes.c2p(1, 1.25))
        p2 = Dot(color=YELLOW).move_to(axes.c2p(3, 3.25))
        secant = Line(p1.get_center(), p2.get_center(), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.add(p1, p2, secant, camera_icon)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Use value tracker for smooth animation
        t = ValueTracker(3)
        def update_p2(m):
            m.move_to(axes.c2p(t.get_value(), 0.25 * t.get_value()**2 + 1))
        def update_secant(m):
            m.put_start_and_end_on(p1.get_center(), p2.get_center())
        
        p2.add_updater(update_p2)
        secant.add_updater(update_secant)
        self.play(t.animate.set_value(1.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        tangent = Line(color=ORANGE).set_length(2).rotate(np.arctan(0.5)).move_to(p1)
        p2.remove_updater(update_p2)
        secant.remove_updater(update_secant)
        self.play(FadeOut(secant), FadeOut(p2), FadeIn(tangent))
        self.wait(2)
