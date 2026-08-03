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
        self.setup_layout("Prerequisite: The Anatomy of a Pure Wave", [
            "Sine waves are defined by height, speed, and phase.",
            "Rotation on a circle generates these fundamental oscillations.",
            "Pure waves of different frequencies never interfere with identity."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Color matching for Line 1
        self.lecture[0].set_color("#FF0000")
        
        # Asset: Circle [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg]
        circle_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        circle_svg.set_color(WHITE).set_stroke(width=2)
        # Fix: Issue 27 - Move circle to 'C3'
        self.place_at_grid(circle_svg, "C3", scale_factor=0.8)
        
        # Logical circle for calculations
        logic_circle = Circle(radius=circle_svg.width/2).move_to(circle_svg.get_center())
        
        time_tracker = ValueTracker(0)
        
        # The dot on the circle (#FF0000)
        dot = Dot(color="#FF0000")
        dot.add_updater(lambda d: d.move_to(logic_circle.point_at_angle(time_tracker.get_value())))
        
        # Sine wave graph area
        # Fix: Issue 28 - Move axes to 'C4' to 'E6'
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3.0,
            y_length=2.0,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(axes, "C4", "E6")
        
        # Tracing point and path
        tracing_point = Dot(color="#FF0000", radius=0.01) # Small point for path tracing
        
        def update_tracing_point(tp):
            t = time_tracker.get_value()
            x_val = (t % 4) 
            y_val = np.sin(t)
            tp.move_to(axes.c2p(x_val, y_val))
        tracing_point.add_updater(update_tracing_point)
        
        # Trace the sine wave
        wave_trail = TracedPath(tracing_point.get_center, stroke_color="#FF0000", stroke_width=4)

        # Persistent connecting line with updater (avoids always_redraw)
        connecting_line = Line(dot.get_center(), tracing_point.get_center(), stroke_width=1, stroke_opacity=0.5, color=WHITE)
        def update_line(l):
            l.put_start_and_end_on(dot.get_center(), tracing_point.get_center())
        connecting_line.add_updater(update_line)

        self.play(FadeIn(circle_svg), Create(axes))
        self.add(dot, tracing_point, wave_trail, connecting_line)
        self.play(time_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        
        # Labels for Amplitude and Frequency
        amp_line = Line(logic_circle.get_center(), logic_circle.get_top(), color="#00FFFF")
        amp_label = Text("Amplitude", font_size=18, color=WHITE)
        # Fix: Issue 27 - Move amp_label to 'B2'
        self.place_at_grid(amp_label, "B2")
        
        freq_label = Text("Frequency (n)", font_size=18, color=WHITE)
        # Fix: Issue 27 - Move freq_label to 'A3'
        self.place_at_grid(freq_label, "A3")
        
        self.play(Create(amp_line), Write(amp_label), Write(freq_label))
        self.play(time_tracker.animate.set_value(8), run_time=4, rate_func=linear)
        self.wait(1)
        
        self.play(
            FadeOut(amp_line), FadeOut(amp_label), FadeOut(freq_label), 
            FadeOut(circle_svg), FadeOut(dot), FadeOut(wave_trail), FadeOut(axes),
            FadeOut(connecting_line), FadeOut(tracing_point)
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Two waves showing different frequencies sin(x), sin(2x)
        axes_comp = Axes(
            x_range=[0, 2*PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(axes_comp, "B2", "E6")
        
        # Asset: Two waves [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg]
        sin1_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg")
        sin1_svg.set_color("#FFFF00").set_stroke(width=4)
        sin1_svg.stretch_to_fit_width(axes_comp.x_length)
        sin1_svg.stretch_to_fit_height(axes_comp.y_length / 2)
        sin1_svg.move_to(axes_comp.c2p(PI, 0))
        
        sin2_part1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg")
        sin2_part2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg")
        sin2_svg = VGroup(sin2_part1, sin2_part2).arrange(RIGHT, buff=0)
        sin2_svg.set_color("#00FFFF").set_stroke(width=4)
        sin2_svg.stretch_to_fit_width(axes_comp.x_length)
        sin2_svg.stretch_to_fit_height(axes_comp.y_length / 2)
        sin2_svg.move_to(axes_comp.c2p(PI, 0))
        
        label1 = Text("sin(x)", color="#FFFF00", font_size=20)
        # Fix: Issue 26 - Move label1 to 'B6'
        self.place_at_grid(label1, "B6")
        
        label2 = Text("sin(2x)", color="#00FFFF", font_size=20)
        # Fix: Issue 26 - Move label2 to 'E6'
        self.place_at_grid(label2, "E6")
        
        self.play(Create(axes_comp))
        self.play(FadeIn(sin1_svg), Write(label1))
        self.play(FadeIn(sin2_svg), Write(label2))
        self.wait(3)
