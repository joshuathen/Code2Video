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
            "Derivatives measure the instantaneous rate of change.",
            "On a graph, this is the tangent's slope.",
            "Let's look at Swiftie's position over time.",
            "A secant line narrows down to one point.",
            "This steepness shows Swiftie's exact speedometer reading."
        ]
        self.setup_layout("The Derivative: Zooming In (Slope)", lecture_lines)
        
        # Initial state: dim all lecture lines
        self.lecture.set_color(GREY_D)

        # === Animation for Lecture Line 1 ===
        # "Derivatives measure the instantaneous rate of change."
        # A white curve (#FFFFFF) appears; a single point P is highlighted.
        self.lecture[0].set_color(WHITE)
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False}
        ).scale(0.8)
        self.place_in_area(axes, "B1", "F6")
        
        def func(x):
            return 0.2 * x**2 + 0.1 * x + 0.5
            
        curve = axes.plot(func, x_range=[0, 4.5], color=WHITE)
        
        p_x = 2
        p_y = func(p_x)
        point_p = Dot(axes.c2p(p_x, p_y), color=WHITE)
        label_p = MathTex("P", font_size=24, color=WHITE).next_to(point_p, DOWN, buff=0.1)
        
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(point_p), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "On a graph, this is the tangent's slope."
        # A yellow tangent line (#FFD700) appears at point P, label "Slope = Velocity".
        self.lecture[0].set_color(GREY_D)
        self.lecture[1].set_color("#FFD700")
        
        slope_p = 0.4 * p_x + 0.1 # f'(x) = 0.4x + 0.1
        tangent_line = Line(
            axes.c2p(p_x - 1.2, p_y - 1.2 * slope_p),
            axes.c2p(p_x + 1.2, p_y + 1.2 * slope_p),
            color="#FFD700"
        )
        slope_label = Text("Slope = Velocity", font_size=18, color="#FFD700")
        # ISSUE 39: Change slope_label placement to B6
        self.place_at_grid(slope_label, "B6", scale_factor=0.8)
        
        self.play(Create(tangent_line), FadeIn(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Let's look at Swiftie's position over time."
        # Graph title "Swiftie's Position vs Time" (#FFFFFF) appears.
        self.lecture[1].set_color(GREY_D)
        self.lecture[2].set_color(WHITE)
        
        graph_title = Text("Swiftie's Position vs Time", font_size=22, color=WHITE)
        # ISSUE 37: Change graph_title placement to area A1-A6
        self.place_in_area(graph_title, "A1", "A6", scale_factor=0.9)
        
        self.play(Write(graph_title))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A secant line narrows down to one point."
        # A secant line through P and a moving point Q rotates.
        self.lecture[2].set_color(GREY_D)
        self.lecture[3].set_color(BLUE)
        
        q_x_tracker = ValueTracker(4)
        
        # Secant line and point Q
        start_q_x = q_x_tracker.get_value()
        secant_line = Line(axes.c2p(p_x, p_y), axes.c2p(start_q_x, func(start_q_x)), color=BLUE)
        point_q = Dot(axes.c2p(start_q_x, func(start_q_x)), color=BLUE)
        
        def update_secant(mob):
            qx = q_x_tracker.get_value()
            qy = func(qx)
            if abs(qx - p_x) < 0.01:
                slope = slope_p
            else:
                slope = (qy - p_y) / (qx - p_x)
            
            # Draw line segments that extend slightly from P and Q
            start = axes.c2p(p_x - 1.2, p_y - 1.2 * slope)
            end = axes.c2p(p_x + 1.2, p_y + 1.2 * slope)
            mob.set_points_as_corners([start, end])
            
        def update_q(mob):
            qx = q_x_tracker.get_value()
            mob.move_to(axes.c2p(qx, func(qx)))

        secant_line.add_updater(update_secant)
        point_q.add_updater(update_q)
        
        self.play(FadeIn(point_q), Create(secant_line))
        self.play(q_x_tracker.animate.set_value(p_x + 0.01), run_time=3)
        self.wait(1)
        
        secant_line.remove_updater(update_secant)
        point_q.remove_updater(update_q)
        self.play(FadeOut(point_q), FadeOut(secant_line))

        # === Animation for Lecture Line 5 ===
        # "This steepness shows Swiftie's exact speedometer reading."
        # Zoom into P, and a speedometer [Asset: ...] shows speed.
        self.lecture[3].set_color(GREY_D)
        self.lecture[4].set_color("#C0C0C0")
        
        # ISSUE 31: Use Asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        speedometer.set_color("#C0C0C0")
        speed_text = Text("Speed", font_size=14, color="#C0C0C0").next_to(speedometer, DOWN, buff=0.1)
        speedo_group = VGroup(speedometer, speed_text)
        
        # ISSUE 38: Place speedometer at F6
        self.place_at_grid(speedo_group, "F6", scale_factor=0.6)
        
        # Group objects for zoom simulation (excluding lecture and title)
        anim_vgroup = VGroup(axes, curve, point_p, label_p, tangent_line, slope_label, graph_title)
        zoom_center = point_p.get_center()
        
        self.play(
            anim_vgroup.animate.scale(1.8, about_point=zoom_center),
            FadeIn(speedo_group),
            run_time=2
        )
        self.wait(2)
