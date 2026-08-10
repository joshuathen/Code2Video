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
        self.setup_layout("The Zooming Transformation (Secant to Tangent)", [
            "Secant line connects two points.",
            "Moving points closer reduces distance.",
            "Zooming reveals a straight line.",
            "Secant transforms into a tangent.",
            "Tangent shows instantaneous slope."
        ])

        # Define Curve: y = 0.5 * x^2
        axes = Axes(x_range=[-2, 2], y_range=[-0.5, 2], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, color=WHITE)
        
        # Grid layout placement
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, 'B4', 'E6', scale_factor=0.45)

        # Points
        p_x, q_x = -1.0, 1.0
        p_point = Dot(axes.c2p(p_x, 0.5 * p_x**2), color="#FF0000")
        q_point = Dot(axes.c2p(q_x, 0.5 * q_x**2), color="#FF0000")
        
        secant = Line(p_point.get_center(), q_point.get_center(), color=WHITE)
        
        # Setup trackers
        q_tracker = ValueTracker(q_x)
        
        def update_secant(l):
            new_q = axes.c2p(q_tracker.get_value(), 0.5 * q_tracker.get_value()**2)
            q_point.move_to(new_q)
            l.put_start_and_end_on(p_point.get_center(), new_q)
            
        secant.add_updater(update_secant)

        # Asset
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        self.place_at_grid(camera_icon, 'A4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(p_point), FadeIn(q_point), Create(secant))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        self.play(q_tracker.animate.set_value(-0.2), run_time=2)
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Zooming/Transformation
        self.play(FadeIn(camera_icon), graph_group.animate.scale(2.0), run_time=2)
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        secant.remove_updater(update_secant)
        tangent = Line(axes.c2p(-1.5, 0.5), axes.c2p(-0.5, 0.5), color="#00FFFF")
        self.play(ReplacementTransform(secant, tangent))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        tangent_label = Text("Tangent", color="#00FFFF", font_size=24)
        self.place_at_grid(tangent_label, 'E4', scale_factor=0.7)
        self.play(Write(tangent_label), FadeOut(camera_icon))
        self.lecture[4].set_color("#00FFFF")

        self.wait(2)
