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
        # Data
        title = "Prerequisite: The Harmony of Sine Waves"
        lines = [
            "Every complex periodic shape is built from simple sine waves.",
            "By adding waves of different frequencies, new shapes emerge.",
            "These building blocks combine to create the harmony we see."
        ]
        self.setup_layout(title, lines)

        # Colors
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#0000FF"
        WHITE_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # Time tracker for synchronized oscillations
        time_tracker = ValueTracker(0)
        self.add(time_tracker)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        # === Animation for Lecture Line 1 ===
        # Description: Show three dots (red, green, blue) oscillating.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        dot_red = Dot(color=RED_COLOR)
        dot_green = Dot(color=GREEN_COLOR)
        dot_blue = Dot(color=BLUE_COLOR)
        
        # Position dots in Column 4 (Rows B, C, D) to avoid lecture overlap
        self.place_at_grid(dot_red, "B4", scale_factor=0.8)
        self.place_at_grid(dot_green, "C4", scale_factor=0.8)
        self.place_at_grid(dot_blue, "D4", scale_factor=0.8)
        
        # Capture baseline positions for updaters
        pos_red = dot_red.get_center().copy()
        pos_green = dot_green.get_center().copy()
        pos_blue = dot_blue.get_center().copy()
        
        # Add updaters for oscillation (vertical)
        dot_red.add_updater(lambda d: d.move_to(pos_red + UP * 0.4 * np.sin(2 * PI * 0.5 * time_tracker.get_value())))
        dot_green.add_updater(lambda d: d.move_to(pos_green + UP * 0.4 * np.sin(2 * PI * 1.0 * time_tracker.get_value())))
        dot_blue.add_updater(lambda d: d.move_to(pos_blue + UP * 0.4 * np.sin(2 * PI * 1.5 * time_tracker.get_value())))
        
        self.play(FadeIn(dot_red), FadeIn(dot_green), FadeIn(dot_blue))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Description: Overlay the three corresponding sine waves.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Create VMobjects for waves
        wave_red = VMobject(color=RED_COLOR)
        wave_green = VMobject(color=GREEN_COLOR)
        wave_blue = VMobject(color=BLUE_COLOR)
        
        # Place waves in areas (Rows B, C, D; Columns 4 to 6) to avoid lecture overlap
        self.place_in_area(wave_red, "B4", "B6", scale_factor=0.9)
        self.place_in_area(wave_green, "C4", "C6", scale_factor=0.9)
        self.place_in_area(wave_blue, "D4", "D6", scale_factor=0.9)
        
        # Baseline centers for the updater logic
        c_red = wave_red.get_center().copy()
        c_green = wave_green.get_center().copy()
        c_blue = wave_blue.get_center().copy()
        
        # Function to generate points for a moving sine wave centered at 'center'
        # Width adjusted to 2.0 to fit Column 4-6
        def get_wave_points(center, freq, amp, t, width=2.0):
            res = 60
            x_vals = np.linspace(-width/2, width/2, res)
            return [center + np.array([x, amp * np.sin(2 * PI * freq * (t - x)), 0]) for x in x_vals]

        # Add updaters to VMobjects to avoid creating new objects each frame
        wave_red.add_updater(lambda m: m.set_points_as_corners(get_wave_points(c_red, 0.5, 0.4, time_tracker.get_value())))
        wave_green.add_updater(lambda m: m.set_points_as_corners(get_wave_points(c_green, 1.0, 0.4, time_tracker.get_value())))
        wave_blue.add_updater(lambda m: m.set_points_as_corners(get_wave_points(c_blue, 1.5, 0.4, time_tracker.get_value())))
        
        self.play(Create(wave_red), Create(wave_green), Create(wave_blue))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Description: Sum the waves into one jagged white path.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Create a VMobject for the combined wave
        sum_wave = VMobject(color=WHITE_COLOR, stroke_width=4)
        # Place in a larger area (Row F, Columns 4 to 6) to avoid lecture overlap
        self.place_in_area(sum_wave, "F4", "F6", scale_factor=0.9)
        c_sum = sum_wave.get_center().copy()
        
        # Function to generate points for the summation of the three sine waves
        # Width adjusted to 2.0 to fit Column 4-6
        def get_sum_points(center, t, width=2.0):
            res = 100
            x_vals = np.linspace(-width/2, width/2, res)
            points = []
            for x in x_vals:
                y = (0.4 * np.sin(2 * PI * 0.5 * (t - x)) + 
                     0.4 * np.sin(2 * PI * 1.0 * (t - x)) + 
                     0.4 * np.sin(2 * PI * 1.5 * (t - x)))
                points.append(center + np.array([x, y, 0]))
            return points
            
        sum_wave.add_updater(lambda m: m.set_points_as_corners(get_sum_points(c_sum, time_tracker.get_value())))
        
        # Label centered above the sum wave (Row E, Columns 4 to 6)
        sum_label = Text("Summed Result", font_size=24, color=WHITE_COLOR)
        self.place_in_area(sum_label, "E4", "E6", scale_factor=0.8)
        
        self.play(Create(sum_wave), Write(sum_label))
        self.wait(4)
