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
        # Setup the layout with the lecture lines for section 5
        self.setup_layout(
            "Measurement: The Great Collapse",
            [
                "Measuring a system forces a single definite outcome.",
                "The blurred superposition instantly collapses into one state.",
                "The vector snaps from a diagonal to an axis.",
                "Measurement is an irreversible change to the quantum state.",
                "Once observed, the system loses its original superposition."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Draw axes
        axes = Axes(
            x_range=[0, 1.5, 1],
            y_range=[0, 1.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": BLUE_E}
        )
        self.place_in_area(axes, "B2", "E5")
        
        label_0 = MathTex(r"|0\rangle", color=WHITE, font_size=28)
        label_1 = MathTex(r"|1\rangle", color=WHITE, font_size=28)
        
        # Position labels using the grid system to ensure no clipping
        self.place_at_grid(label_0, "E6", scale_factor=0.8)
        self.place_at_grid(label_1, "A2", scale_factor=0.8)
        
        # Create the superposition vector at 45 degrees
        vec_start = axes.c2p(0, 0)
        vec_end = axes.c2p(1, 1)
        psi_vec = Arrow(vec_start, vec_end, buff=0, color="#00FF00", stroke_width=4)
        psi_label = MathTex(r"|\psi\rangle", color="#00FF00", font_size=28)
        psi_label.next_to(vec_end, UR, buff=0.1)
        
        self.play(Create(axes), Create(label_0), Create(label_1))
        self.play(GrowArrow(psi_vec), FadeIn(psi_label))
        
        # Pulsing blur effect (ValueTracker + updater)
        pulse_tracker = ValueTracker(0)
        def pulse_func(m):
            m.set_opacity(0.6 + 0.3 * np.sin(pulse_tracker.get_value() * PI * 2))
        
        psi_vec.add_updater(pulse_func)
        
        self.play(pulse_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create an eye icon (Measurement)
        eye_outer = Ellipse(width=1.2, height=0.6, color=WHITE)
        eye_iris = Circle(radius=0.25, color=BLUE_B, fill_opacity=1)
        eye_pupil = Circle(radius=0.1, color=BLACK, fill_opacity=1)
        eye = VGroup(eye_outer, eye_iris, eye_pupil)
        
        # Fix for Issue 36: Move eye from A3 to A4
        self.place_at_grid(eye, "A4", scale_factor=0.8)
        
        self.play(FadeIn(eye, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Remove pulsing updater before the snap
        psi_vec.remove_updater(pulse_func)
        
        # Target position: snapped to |0> axis (horizontal)
        target_vec_end = axes.c2p(1, 0)
        
        # Snap vector and change color to white
        # Use rush_from for a "snappy" start
        self.play(
            psi_vec.animate.set_color(WHITE).set_opacity(1.0).put_start_and_end_on(vec_start, target_vec_end),
            psi_label.animate.next_to(target_vec_end, DOWN, buff=0.2).set_color(WHITE),
            run_time=0.4,
            rate_func=rush_from
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Fade away |1> axis and its label to show irreversibility
        self.play(FadeOut(label_1), FadeOut(axes.y_axis))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # "The Collapse" text in bright red with a quick flash
        collapse_text = Text("The Collapse", color="#FF0000", font_size=40, weight=BOLD)
        # Fix for Issue 35: Move collapse_text from D3-D4 to D5-E6
        self.place_in_area(collapse_text, "D5", "E6")
        
        flash_rect = Rectangle(width=20, height=20, fill_color=WHITE, fill_opacity=0.8, stroke_width=0)
        flash_rect.set_z_index(10)
        
        self.play(
            FadeIn(flash_rect, run_time=0.1),
            Write(collapse_text, run_time=0.2),
        )
        self.play(FadeOut(flash_rect, run_time=0.3))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
        self.wait(1)
