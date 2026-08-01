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
        # Initial Setup
        lines = [
            'The sine wave is our simplest building block.',
            'Amplitude shows strength, while frequency shows speed.',
            'A rotating point perfectly traces this smooth wave.'
        ]
        self.setup_layout("Prerequisite: The Pure Tone (Sine Waves)", lines)

        # Colors for matching lecture lines
        COLOR_LINE1 = "#FFFFFF" # White
        COLOR_LINE2 = "#FFFF00" # Yellow
        COLOR_LINE3 = "#0000FF" # Blue

        # Assets
        CIRCLE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/circle.svg"
        RADIUS_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/radius.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LINE1)
        
        # Load and place the circle asset [Asset: circle.svg]
        circle = SVGMobject(CIRCLE_ASSET).set_color(COLOR_LINE1)
        circle.height = 1.6 
        self.place_at_grid(circle, "C2", scale_factor=0.8) # Issue 34: Scaling circle
        circle_center = circle.get_center()
        radius_val = circle.height / 2
        
        # Value Tracker for rotation angle
        theta = ValueTracker(0)
        
        # Load and place the radius line asset [Asset: radius.svg]
        radius_line = SVGMobject(RADIUS_ASSET).set_color(COLOR_LINE1)
        radius_line.width = radius_val
        radius_line.move_to(circle_center + RIGHT * (radius_val / 2))
        
        # Dot on the circle that follows the radius end
        dot_on_circle = Dot(color=COLOR_LINE3)
        dot_on_circle.add_updater(lambda d: d.move_to(
            circle_center + np.array([radius_val * np.cos(theta.get_value()), radius_val * np.sin(theta.get_value()), 0])
        ))

        # Radius line rotation updater using a closure to track delta theta
        angle_tracker = [0]
        def update_radius(rl):
            curr_theta = theta.get_value()
            rl.rotate(curr_theta - angle_tracker[0], about_point=circle_center)
            angle_tracker[0] = curr_theta
        
        radius_line.add_updater(update_radius)

        self.play(Create(circle), FadeIn(radius_line), FadeIn(dot_on_circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_LINE2)
        
        # Amplitude visualization (vertical distance)
        amp_arrow = DoubleArrow(
            start=circle_center + LEFT * 1.0, 
            end=circle_center + LEFT * 1.0 + UP * radius_val,
            buff=0, color=COLOR_LINE2, stroke_width=3
        )
        amp_label = Text("Amplitude", font_size=16, color=COLOR_LINE2)
        self.place_at_grid(amp_label, "B1", scale_factor=0.7) # Issue 33: Label position

        # Frequency visualization (horizontal spacing concept)
        freq_arrow = DoubleArrow(
            start=self.grid["D3"] + LEFT * 0.5,
            end=self.grid["D4"] + LEFT * 0.5,
            buff=0, color=COLOR_LINE2, stroke_width=3
        )
        freq_label = Text("Frequency (spacing)", font_size=16, color=COLOR_LINE2)
        self.place_in_area(freq_label, "E2", "E4", scale_factor=0.6) # Issue 32: Label position

        self.play(GrowFromCenter(amp_arrow), Write(amp_label))
        self.play(GrowFromCenter(freq_arrow), Write(freq_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_LINE3)
        
        # Sine wave tracing setup (starting from right of circle area)
        wave_start_x = self.grid["C3"][0] - 0.5
        
        # Traced path for the blue sine wave (#0000FF)
        sine_path = VMobject(color=COLOR_LINE3)
        sine_path.set_points_as_corners([[wave_start_x, circle_center[1], 0]])
        
        def update_path(path):
            current_theta = theta.get_value()
            points = []
            step = 0.1
            for t in np.arange(0, current_theta + step, step):
                # x grows to the right, y oscillates based on circle radius
                x_pos = wave_start_x + t * 0.4
                y_pos = circle_center[1] + radius_val * np.sin(t)
                points.append([x_pos, y_pos, 0])
            
            if len(points) > 1:
                path.set_points_as_corners(points)

        sine_path.add_updater(update_path)
        
        # A horizontal connecting line from circle dot to wave dot
        connector = Line(color=GRAY, stroke_opacity=0.5)
        connector.add_updater(lambda l: l.set_points_as_corners([
            dot_on_circle.get_center(),
            [wave_start_x + theta.get_value() * 0.4, dot_on_circle.get_center()[1], 0]
        ]))

        # Moving dot on the wave following the trace
        dot_on_wave = Dot(color=COLOR_LINE3)
        dot_on_wave.add_updater(lambda d: d.move_to([
            wave_start_x + theta.get_value() * 0.4, 
            circle_center[1] + radius_val * np.sin(theta.get_value()), 
            0
        ]))

        self.add(sine_path, connector, dot_on_wave)
        
        # Animate the rotation and tracing (about 2 full rotations)
        self.play(theta.animate.set_value(4 * PI), run_time=6, rate_func=linear)
        self.wait(2)
