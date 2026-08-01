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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with lines
        lecture_lines = [
            'Every material has a natural resonant frequency.',
            'Light frequency determines how strongly it interacts.',
            'High-frequency blue light interacts more than red light.',
            'Stronger interactions lead to a higher refractive index.',
            'This frequency dependence is called dispersion.'
        ]
        self.setup_layout("Why Color Matters: The Frequency Connection", lecture_lines)

        # Common Colors
        COLOR_ORANGE = "#FF4500"
        COLOR_RED = "#FF0000"
        COLOR_VIOLET = "#EE82EE"
        COLOR_WHITE = "#FFFFFF"

        # Time tracker for animations
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ORANGE)
        
        # Atom structure: Nucleus and Electron with Spring
        nucleus = Dot(color=GREY, radius=0.15)
        electron = Dot(color=BLUE, radius=0.1)
        self.place_at_grid(nucleus, "B3")
        
        # Spring representation
        def get_spring(mob_n, mob_e):
            n_pos = mob_n.get_center()
            e_pos = mob_e.get_center()
            dist = np.linalg.norm(e_pos - n_pos)
            if dist < 0.01: return Line(n_pos, e_pos)
            
            vec = (e_pos - n_pos) / dist
            perp = np.array([-vec[1], vec[0], 0])
            points = []
            steps = 20
            for i in range(steps + 1):
                p = n_pos + (i / steps) * (e_pos - n_pos)
                if 0 < i < steps:
                    p += 0.1 * np.sin(i * PI) * perp
                points.append(p)
            return VMobject().set_points_as_corners(points).set_color(WHITE).set_stroke(width=2)

        # Electron oscillation setup
        electron.add_updater(lambda m: m.move_to(nucleus.get_center() + np.array([0, 0.4 * np.sin(3 * time_tracker.get_value()), 0])))
        spring = always_redraw(lambda: get_spring(nucleus, electron))

        # Resonance Meter
        meter_frame = Rectangle(width=2, height=0.4, color=WHITE)
        self.place_in_area(meter_frame, "A3", "A5")
        meter_label = Text("Resonance", font_size=16, color=COLOR_ORANGE).next_to(meter_frame, UP, buff=0.1)
        meter_fill = Rectangle(width=0, height=0.3, fill_opacity=0.8, color=COLOR_ORANGE, stroke_width=0)
        meter_fill.align_to(meter_frame, LEFT).shift(RIGHT * 0.05)

        self.play(FadeIn(nucleus), FadeIn(electron), Create(spring), Create(meter_frame), Write(meter_label))
        self.play(meter_fill.animate.set_width(1.8).align_to(meter_frame, LEFT).shift(RIGHT * 0.05), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Clear previous to make space for 2-atom comparison
        self.play(FadeOut(nucleus), FadeOut(electron), FadeOut(spring), FadeOut(meter_frame), FadeOut(meter_label), FadeOut(meter_fill))

        # Two comparison setups: Red vs Violet
        # Atoms
        n1 = Dot(color=GREY, radius=0.1); self.place_at_grid(n1, "C3")
        e1 = Dot(color=BLUE, radius=0.07)
        n2 = Dot(color=GREY, radius=0.1); self.place_at_grid(n2, "E3")
        e2 = Dot(color=BLUE, radius=0.07)

        # Waves approaching
        def wave_func(x, t, freq, color):
            return FunctionGraph(lambda x_val: 0.3 * np.sin(freq * (x_val - 0.5*t)), x_range=[-1.5, 0], color=color).shift(RIGHT * 0.5)

        red_wave_init = wave_func(0, 0, 4, COLOR_RED); self.place_at_grid(red_wave_init, "C1")
        violet_wave_init = wave_func(0, 0, 10, COLOR_VIOLET); self.place_at_grid(violet_wave_init, "E1")

        self.play(FadeIn(n1, e1, n2, e2), Create(red_wave_init), Create(violet_wave_init))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_VIOLET)

        # Animate interaction (updaters)
        e1.add_updater(lambda m: m.move_to(n1.get_center() + np.array([0, 0.15 * np.sin(4 * (time_tracker.get_value())), 0])))
        e2.add_updater(lambda m: m.move_to(n2.get_center() + np.array([0, 0.6 * np.sin(10 * (time_tracker.get_value())), 0])))
        
        spring1 = always_redraw(lambda: get_spring(n1, e1))
        spring2 = always_redraw(lambda: get_spring(n2, e2))
        self.add(spring1, spring2)

        # Replace static waves with moving ones
        red_wave_moving = always_redraw(lambda: wave_func(0, 4*time_tracker.get_value(), 4, COLOR_RED).move_to(self.grid["C1"]))
        violet_wave_moving = always_redraw(lambda: wave_func(0, 4*time_tracker.get_value(), 10, COLOR_VIOLET).move_to(self.grid["E1"]))
        
        self.remove(red_wave_init, violet_wave_init)
        self.add(red_wave_moving, violet_wave_moving)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)

        # Phase shift vectors (Section 3 style)
        circle_red = Circle(radius=0.4, color=WHITE).move_to(self.grid["C5"])
        circle_violet = Circle(radius=0.4, color=WHITE).move_to(self.grid["E5"])
        
        vec_red_incident = Arrow(circle_red.get_center(), circle_red.get_center() + RIGHT*0.4, buff=0, color=WHITE)
        vec_red_shift = Arrow(circle_red.get_center(), circle_red.get_center() + rotate_vector(RIGHT*0.4, -20*DEGREES), buff=0, color=COLOR_RED)
        
        vec_violet_incident = Arrow(circle_violet.get_center(), circle_violet.get_center() + RIGHT*0.4, buff=0, color=WHITE)
        vec_violet_shift = Arrow(circle_violet.get_center(), circle_violet.get_center() + rotate_vector(RIGHT*0.4, -70*DEGREES), buff=0, color=COLOR_VIOLET)

        self.play(Create(circle_red), Create(circle_violet))
        self.play(GrowArrow(vec_red_incident), GrowArrow(vec_violet_incident))
        self.play(GrowArrow(vec_red_shift), GrowArrow(vec_violet_shift))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_WHITE)

        # Dispersion: Clean up and show material block
        self.play(FadeOut(VGroup(n1, e1, n2, e2, spring1, spring2, red_wave_moving, violet_wave_moving, circle_red, circle_violet, vec_red_incident, vec_red_shift, vec_violet_incident, vec_violet_shift)))

        material_box = Rectangle(width=3, height=2, fill_opacity=0.2, color=BLUE)
        self.place_in_area(material_box, "C2", "E5")
        dispersion_text = Text("Dispersion", color=COLOR_WHITE, font_size=24)
        self.place_at_grid(dispersion_text, "B4")

        # Visualizing speed difference
        # Dot representing red light (faster)
        red_dot = Dot(color=COLOR_RED).move_to(self.grid["C1"])
        violet_dot = Dot(color=COLOR_VIOLET).move_to(self.grid["D1"])
        
        path_start = self.grid["C1"]
        path_end = self.grid["C6"]
        
        self.play(FadeIn(material_box), Write(dispersion_text))
        self.play(FadeIn(red_dot, violet_dot))
        
        # Red moves faster through medium than violet
        # Vacuum speed: 2 units/s, Red speed: 1.5 units/s, Violet speed: 1 unit/s
        self.play(
            red_dot.animate.move_to(self.grid["C6"]),
            violet_dot.animate.move_to(self.grid["D6"]),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
