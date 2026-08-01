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
        # Setup metadata
        title = "The Winding Machine (Visual Intuition)"
        lines = [
            "Let's wrap our signal around a circular pole.",
            "At most frequencies, the result is a chaotic scribble.",
            "But at the magic frequency, a pattern emerges.",
            "The center of mass shifts far from the origin.",
            "This shift marks a hidden component of the signal."
        ]
        self.setup_layout(title, lines)

        # Variables
        f_sig = 2.0  # Fixed signal frequency (Hz)
        f_wind = ValueTracker(0.8)  # Initial winding frequency (Hz)
        t_max = 5.0  # Duration of the signal segment
        
        # Color definitions
        CIRCLE_COLOR = "#FFFFFF"
        SCRIBBLE_COLOR = "#888888"
        COM_COLOR = "#FFFF00"
        SPIKE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CIRCLE_COLOR)
        
        # Create circular pole and rotating vector
        pole = Circle(radius=1.5, color=CIRCLE_COLOR, stroke_opacity=0.3)
        self.place_in_area(pole, 'A2', 'D5', scale_factor=0.9) # Issue 32 fix
        
        # Machine asset integration (Issue 26)
        machine = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/machine.svg")
        self.place_at_grid(machine, "A1", scale_factor=0.5)
        
        rotating_vector = Arrow(pole.get_center(), pole.get_center() + RIGHT * 1.5, buff=0, color=CIRCLE_COLOR)
        rotating_vector.add_updater(lambda m, dt: m.rotate(f_wind.get_value() * TAU * dt, about_point=pole.get_center()))
        
        self.play(Create(pole), FadeIn(machine), GrowArrow(rotating_vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(SCRIBBLE_COLOR)

        def get_scribble_points(freq):
            points = []
            steps = 250
            for i in range(steps):
                t = (i / steps) * t_max
                # Signal: g(t) = 1 + cos(2 * pi * f_sig * t)
                r = 1 + np.cos(TAU * f_sig * t)
                # Wrapping: r * e^(-i * 2 * pi * f_wind * t)
                angle = -TAU * freq * t
                points.append(pole.get_center() + np.array([r * np.cos(angle), r * np.sin(angle), 0]) * 0.7)
            return points

        scribble = VMobject(color=SCRIBBLE_COLOR)
        scribble.set_points_as_corners(get_scribble_points(f_wind.get_value()))
        scribble.add_updater(lambda m: m.set_points_as_corners(get_scribble_points(f_wind.get_value())))

        self.play(Create(scribble))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COM_COLOR)

        def get_com_pos():
            pts = scribble.get_all_points()
            if len(pts) == 0: return pole.get_center()
            return np.mean(pts, axis=0)

        com_dot = Dot(color=COM_COLOR)
        com_dot.add_updater(lambda m: m.move_to(get_com_pos()))
        
        com_label = Text("Center of Mass", font_size=16, color=COM_COLOR)
        self.place_at_grid(com_label, 'B4', scale_factor=0.6) # Issue 30 fix

        self.play(FadeIn(com_dot), FadeIn(com_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COM_COLOR)

        # Adjust winding frequency to match signal frequency (2.0)
        self.play(f_wind.animate.set_value(2.0), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(SPIKE_COLOR)

        # Setup a small frequency-magnitude graph in the corner
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1, 0.5],
            x_length=2.5,
            y_length=1.5,
            axis_config={"color": WHITE, "include_tip": False},
        )
        self.place_in_area(axes, 'E2', 'F5', scale_factor=0.8) # Issue 31 fix
        
        x_label_obj = Text("f", font_size=16)
        y_label_obj = Text("CM", font_size=16)
        axes_labels = axes.get_axis_labels(x_label=x_label_obj, y_label=y_label_obj)
        
        # The spike at f=2.0
        spike = axes.get_vertical_line(axes.c2p(2.0, 0.8), color=SPIKE_COLOR, stroke_width=4)
        spike_dot = Dot(axes.c2p(2.0, 0.8), color=SPIKE_COLOR)
        
        # Pole asset integration (Issue 26)
        pole_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/pole.svg")
        self.place_at_grid(pole_icon, "F6", scale_factor=0.4)

        self.play(Create(axes), FadeIn(axes_labels))
        self.play(Create(spike), FadeIn(spike_dot), FadeIn(pole_icon))
        
        # Highlight shift one last time
        self.play(Indicate(com_dot), Indicate(spike))
        self.wait(2)
