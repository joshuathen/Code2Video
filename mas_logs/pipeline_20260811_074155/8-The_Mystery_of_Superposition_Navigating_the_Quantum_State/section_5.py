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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Measuring a quantum system forces a definite choice.",
            "The superposition \"collapses\" into one single base state.",
            "Amplitudes determine the likelihood of each specific outcome.",
            "Square the coefficient to find the probability of collapse.",
            "Once observed, the \"blur\" of possibilities instantly vanishes."
        ]
        
        self.setup_layout("The Collapse: Observation and Probability", lecture_lines)
        
        # Define colors
        COLOR_MEASURE = WHITE
        COLOR_COLLAPSE = "#00FF00"  # Green
        COLOR_SPHERE = BLUE_E
        
        # === Animation for Lecture Line 1 ===
        # Show a vector on the equator of the Bloch sphere vibrating rapidly.
        sphere_radius = 1.5
        sphere_circle = Circle(radius=sphere_radius, color=COLOR_SPHERE).set_stroke(opacity=0.5)
        sphere_bg = Circle(radius=sphere_radius, color=COLOR_SPHERE, fill_opacity=0.1)
        self.place_in_area(sphere_circle, "B2", "E5")
        self.place_in_area(sphere_bg, "B2", "E5")
        
        vector_origin = sphere_circle.get_center()
        horiz_axis = DashedLine(vector_origin + LEFT * sphere_radius, vector_origin + RIGHT * sphere_radius, color=GRAY)
        vert_axis = Line(vector_origin + DOWN * sphere_radius, vector_origin + UP * sphere_radius, color=GRAY)
        
        # Vector starts on the equator
        vector = Arrow(vector_origin, vector_origin + RIGHT * sphere_radius, buff=0, color=WHITE)
        
        self.lecture[0].set_color(WHITE)
        self.play(
            FadeIn(sphere_circle), 
            FadeIn(sphere_bg), 
            Create(horiz_axis), 
            Create(vert_axis), 
            GrowArrow(vector)
        )
        
        # Vibration effect using ValueTracker for angle jitter
        vibe_tracker = ValueTracker(0)
        def vibrate_update(m):
            angle = 0.1 * np.sin(vibe_tracker.get_value() * 50)
            target_end = vector_origin + np.array([np.cos(angle), np.sin(angle), 0]) * sphere_radius
            m.put_start_and_end_on(vector_origin, target_end)
            
        vector.add_updater(vibrate_update)
        self.play(vibe_tracker.animate.set_value(1), run_time=1.5, rate_func=linear)
        vector.remove_updater(vibrate_update)
        # Ensure it ends at exactly 0 before next step
        self.play(vector.animate.put_start_and_end_on(vector_origin, vector_origin + RIGHT * sphere_radius), run_time=0.1)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Display a white eye icon (#FFFFFF) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg] representing 'Measurement'.
        self.lecture[1].set_color(COLOR_MEASURE)
        
        # Load asset
        eye_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color=WHITE)
        # Center eye_group at 'A3'-'A4'
        self.place_in_area(eye_icon, "A3", "A4", scale_factor=0.6)
        
        self.play(FadeIn(eye_icon, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The equatorial vector instantly snaps to the North Pole in green (#00FF00).
        self.lecture[2].set_color(COLOR_COLLAPSE)
        
        target_pole_end = vector_origin + UP * sphere_radius
        self.play(
            vector.animate.set_color(COLOR_COLLAPSE).put_start_and_end_on(vector_origin, target_pole_end),
            run_time=0.4,
            rate_func=rush_into
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the text 'P(0) = |α|²' in green (#00FF00) near the pole.
        # Move prob_label to 'A5'-'A6' (scale 0.8) to clear the sphere area.
        self.lecture[3].set_color(COLOR_COLLAPSE)
        
        prob_label = MathTex("P(0) = |\\alpha|^2", color=COLOR_COLLAPSE, font_size=32)
        self.place_in_area(prob_label, "A5", "A6", scale_factor=0.8)
        
        self.play(Write(prob_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Fade out the rest of the sphere, leaving only the North Pole highlighted.
        self.lecture[4].set_color(WHITE)
        
        # Highlight pole dot
        pole_dot = Dot(vector_origin + UP * sphere_radius, color=COLOR_COLLAPSE).scale(1.2)
        
        fade_out_group = VGroup(sphere_circle, sphere_bg, horiz_axis, vert_axis, eye_icon)
        
        self.play(
            fade_out_group.animate.set_opacity(0.15),
            FadeIn(pole_dot),
            run_time=1.5
        )
        self.wait(2)
