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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title_text = "Visual Summary & Real-World Application"
        lecture_lines = [
            "The derivative is the slope of the tangent line.",
            "It measures how things change at every instant.",
            "From physics to economics, calculus is everywhere."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        color1 = YELLOW
        color2 = GREEN
        color3 = "#ADD8E6"  # Light Blue
        
        # Assets
        speedometer_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg"
        
        # Trackers
        # Derivative of 0.5x^2 + 1 is x. Slope varies from -1.5 to 1.5.
        t_tracker = ValueTracker(-1.5)

        # === Animation for Lecture Line 1 ===
        # Create Graph elements (Right area: A4 to F6)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": False},
            x_length=3,
            y_length=3
        ).set_color(WHITE)
        
        func = axes.plot(lambda x: 0.5 * x**2 + 1, color=WHITE)
        
        dot = Dot(color=color1)
        dot.add_updater(lambda m: m.move_to(axes.c2p(t_tracker.get_value(), 0.5 * t_tracker.get_value()**2 + 1)))
        
        tangent_line = Line(color=color1, stroke_width=4)
        def update_tangent(mob):
            val = t_tracker.get_value()
            slope = val 
            y0 = 0.5 * val**2 + 1
            # Length of tangent line segment
            dx = 0.7
            p1 = axes.c2p(val - dx, -dx * slope + y0)
            p2 = axes.c2p(val + dx, dx * slope + y0)
            mob.set_points_as_corners([p1, p2])
        tangent_line.add_updater(update_tangent)
        
        graph_group = VGroup(axes, func, tangent_line, dot)
        self.place_in_area(graph_group, "A4", "F6", scale_factor=0.9)
        
        # Create Speedometer using Asset
        speedometer_svg = SVGMobject(speedometer_path).set_color(WHITE)
        self.place_in_area(speedometer_svg, "A1", "F3", scale_factor=1.2)
        
        # Needle for the speedometer SVG
        # Position needle at the visual center of the gauge
        speed_center = speedometer_svg.get_center() + DOWN * 0.2
        needle = Line(speed_center, speed_center + LEFT * 0.8, color=color2, stroke_width=6)
        
        def update_needle(mob):
            slope = t_tracker.get_value()
            # Slope range is [-1.5, 1.5]. 
            # We map this to the gauge's arc. 
            # -1.5 -> Left (~180 degrees or PI), 1.5 -> Right (~0 degrees or 0)
            norm = (slope + 1.5) / 3.0
            angle = PI - (norm * PI)
            length = 0.8
            p_end = speed_center + np.array([length * np.cos(angle), length * np.sin(angle), 0])
            mob.set_points_as_corners([speed_center, p_end])
            
        needle.add_updater(update_needle)

        # Labels for split screen
        label_speed = Text("Speedometer", font_size=20, color=WHITE)
        label_graph = Text("Tangent Slope", font_size=20, color=WHITE)
        self.place_at_grid(label_speed, "A2", scale_factor=0.8)
        self.place_at_grid(label_graph, "A5", scale_factor=0.8)

        # Execution Line 1: Split-screen view
        self.play(self.lecture[0].animate.set_color(color1))
        self.play(
            Create(axes), Create(func), 
            FadeIn(speedometer_svg),
            Write(label_speed), Write(label_graph)
        )
        self.play(Create(tangent_line), FadeIn(dot), Create(needle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color2)
        )
        
        # Synchronized movement
        self.play(t_tracker.animate.set_value(1.5), run_time=3, rate_func=linear)
        self.play(t_tracker.animate.set_value(-1.5), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color3)
        )
        
        # Broad applicability labels (Positions from issues 36, 37, 38)
        physics_txt = Text("Physics", color=color3, font_size=24)
        econ_txt = Text("Economics", color=color3, font_size=24)
        bio_txt = Text("Biology", color=color3, font_size=24)
        
        # Applying positioning and scaling as requested in issues 36, 37, 38
        self.place_at_grid(physics_txt, "F1", scale_factor=0.8)
        self.place_at_grid(econ_txt, "F3", scale_factor=0.8)
        self.place_at_grid(bio_txt, "F5", scale_factor=0.8)
        
        self.play(
            Write(physics_txt),
            Write(econ_txt),
            Write(bio_txt)
        )
        
        # Final movement
        self.play(t_tracker.animate.set_value(1.5), run_time=4, rate_func=smooth)
        self.wait(2)
