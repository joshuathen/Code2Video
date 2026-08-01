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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'This formula powers modern physics and engineering.',
            'It describes waves, electricity, and quantum mechanics.',
            'A simple circle explains the harmony of our universe.'
        ]
        self.setup_layout("Real-World Echoes", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Colors
        COLOR_VEC = YELLOW
        COLOR_WAVE = "#55FFFF"
        COLOR_ICON = WHITE

        # Grid locations
        plane_center = self.grid["D2"]
        wave_start_x = self.grid["D3"][0]
        wave_end_x = self.grid["D6"][0]

        # Time tracking
        time_tracker = ValueTracker(0)
        drawing_tracker = ValueTracker(0) # Progress of the sine wave trace (0 to width)

        # Complex Plane Elements
        circle = Circle(radius=0.8, color=WHITE)
        self.place_at_grid(circle, "D2")
        
        h_axis_plane = Line(plane_center + LEFT*1.0, plane_center + RIGHT*1.0, color=GRAY, stroke_width=1)
        v_axis_plane = Line(plane_center + DOWN*1.0, plane_center + UP*1.0, color=GRAY, stroke_width=1)

        # Rotating Vector and Dot
        # Using persistent mobjects with updaters
        vector = Line(plane_center, plane_center + RIGHT*0.8, color=COLOR_VEC)
        vector.add_updater(lambda v: v.set_angle(time_tracker.get_value()))
        
        dot_on_circle = Dot(color=COLOR_VEC, radius=0.06)
        dot_on_circle.add_updater(lambda d: d.move_to(vector.get_end()))

        # Display elements
        self.lecture[0].set_color(COLOR_VEC)
        self.play(
            Create(h_axis_plane), 
            Create(v_axis_plane), 
            Create(circle), 
            Create(vector), 
            FadeIn(dot_on_circle)
        )
        self.play(time_tracker.animate.set_value(TAU), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 2 ===
        # Time Axis for Sine Wave
        time_axis = Line(self.grid["D3"], self.grid["D6"], color=GRAY, stroke_width=1)
        
        # Horizontal trace connection (Dashed line)
        connection_line = DashedLine(
            dot_on_circle.get_center(),
            [wave_start_x, dot_on_circle.get_center()[1], 0],
            color=GRAY,
            dash_length=0.05,
            stroke_opacity=0.6
        )
        connection_line.add_updater(lambda l: l.put_start_and_end_on(
            dot_on_circle.get_center(),
            [wave_start_x, dot_on_circle.get_center()[1], 0]
        ))

        # Sine wave via ParametricFunction (efficient for scrolling)
        freq = 3.0
        sine_wave = always_redraw(lambda: ParametricFunction(
            lambda x: np.array([
                x, 
                plane_center[1] + 0.8 * np.sin(time_tracker.get_value() - freq * (x - wave_start_x)), 
                0
            ]),
            t_range=[wave_start_x, wave_start_x + drawing_tracker.get_value()],
            color=COLOR_WAVE
        ))

        self.lecture[1].set_color(COLOR_WAVE)
        self.play(Create(time_axis), Create(connection_line), Create(sine_wave))
        
        # Grow wave trace and continue rotation
        self.play(
            drawing_tracker.animate.set_value(wave_end_x - wave_start_x),
            time_tracker.animate.increment_value(2*TAU),
            run_time=4,
            rate_func=linear
        )

        # === Animation for Lecture Line 3 ===
        # Application Icons using provided assets
        # Signal Tower Icon
        tower = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/tower.svg", color=COLOR_ICON)
        self.place_at_grid(tower, "A4", scale_factor=0.6)

        # Musical Note Icon
        musical_note = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/note.svg", color=COLOR_ICON)
        self.place_at_grid(musical_note, "A6", scale_factor=0.6)

        self.lecture[2].set_color(WHITE)
        self.play(
            FadeIn(tower, shift=UP*0.3),
            FadeIn(musical_note, shift=UP*0.3)
        )
        
        # Final rotation loop to wrap up
        self.play(time_tracker.animate.increment_value(3*TAU), run_time=5, rate_func=linear)
        self.wait(2)
