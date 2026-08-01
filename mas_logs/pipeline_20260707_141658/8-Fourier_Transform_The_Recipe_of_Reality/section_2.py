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
        # Fetching titles and lines from storyboard
        title_text = "Prerequisites: The DNA of Waves"
        lecture_lines = [
            "- Every complex wave consists of simple sine waves.",
            "- Amplitude determines the wave's height or volume.",
            "- Frequency measures how fast the wave oscillates.",
            "- Phase indicates the wave's starting position.",
            "- Circular motion generates these smooth periodic oscillations."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard and constraints
        COLOR_COMPLEX = WHITE
        COLOR_AMP = "#00FF00"
        COLOR_FREQ = "#0000FF"
        COLOR_PHASE = "#FF00FF"
        COLOR_CIRC = "#FFFF00"

        # Shared Axes for wave property demos
        # Updated positioning based on Issue 24: use B3 to E6 to avoid crowding lecture notes
        ax = Axes(
            x_range=[0, 2*PI, PI/2], 
            y_range=[-2.5, 2.5, 1], 
            x_length=4.0, 
            y_length=3.0, 
            axis_config={"include_tip": False}
        ).set_stroke(opacity=0.3)
        self.place_in_area(ax, 'B3', 'E6')

        # === Animation for Lecture Line 1 ===
        # Every complex wave consists of simple sine waves.
        self.lecture[0].set_color(COLOR_COMPLEX)
        
        def wave1_func(x): return np.sin(x)
        def wave2_func(x): return 0.5 * np.sin(3*x)
        def wave3_func(x): return 0.3 * np.sin(5*x)
        def complex_wave_func(x): return wave1_func(x) + wave2_func(x) + wave3_func(x)

        comp_plot = ax.plot(complex_wave_func, color=WHITE)
        
        self.play(Create(ax), Create(comp_plot))
        self.wait(1)

        # Decomposition visualization
        w1_plot = ax.plot(wave1_func, color=WHITE).set_stroke(opacity=0.5)
        w2_plot = ax.plot(wave2_func, color=WHITE).set_stroke(opacity=0.5)
        w3_plot = ax.plot(wave3_func, color=WHITE).set_stroke(opacity=0.5)
        
        self.play(
            FadeIn(w1_plot, shift=DOWN*0.5),
            FadeIn(w2_plot, shift=DOWN*1.0),
            FadeIn(w3_plot, shift=DOWN*1.5),
            comp_plot.animate.set_stroke(opacity=0.2)
        )
        self.wait(2)
        self.play(FadeOut(w1_plot, w2_plot, w3_plot, comp_plot))

        # === Animation for Lecture Line 2 ===
        # Amplitude determines the wave's height or volume.
        self.lecture[1].set_color(COLOR_AMP)
        amp_tracker = ValueTracker(1.0)
        
        # Persistent plot updated by tracker
        amp_sine = always_redraw(lambda: ax.plot(
            lambda x: amp_tracker.get_value() * np.sin(x), 
            color=COLOR_AMP
        ))
        
        # Fixed positioning based on Issue 23: Move to A4 to avoid overlap
        amp_label = Text("Amplitude", font_size=24, color=COLOR_AMP)
        self.place_at_grid(amp_label, 'A4')
        
        self.play(Create(amp_sine), Write(amp_label))
        self.play(amp_tracker.animate.set_value(2.2), run_time=1.5)
        self.play(amp_tracker.animate.set_value(0.5), run_time=1.5)
        self.play(amp_tracker.animate.set_value(1.0), run_time=1)
        self.wait(1)
        self.play(FadeOut(amp_label))

        # === Animation for Lecture Line 3 ===
        # Frequency measures how fast the wave oscillates.
        self.lecture[2].set_color(COLOR_FREQ)
        freq_tracker = ValueTracker(1.0)
        
        freq_sine = always_redraw(lambda: ax.plot(
            lambda x: np.sin(freq_tracker.get_value() * x), 
            color=COLOR_FREQ
        ))
        
        # Fixed positioning based on Issue 23: Move to A4
        freq_label = Text("Frequency", font_size=24, color=COLOR_FREQ)
        self.place_at_grid(freq_label, 'A4')
        
        self.play(ReplacementTransform(amp_sine, freq_sine), Write(freq_label))
        self.play(freq_tracker.animate.set_value(4.0), run_time=1.5)
        self.play(freq_tracker.animate.set_value(0.5), run_time=1.5)
        self.play(freq_tracker.animate.set_value(1.0), run_time=1)
        self.wait(1)
        self.play(FadeOut(freq_label))

        # === Animation for Lecture Line 4 ===
        # Phase indicates the wave's starting position.
        self.lecture[3].set_color(COLOR_PHASE)
        phase_tracker = ValueTracker(0.0)
        
        phase_sine = always_redraw(lambda: ax.plot(
            lambda x: np.sin(x + phase_tracker.get_value()), 
            color=COLOR_PHASE
        ))
        
        # Fixed positioning based on Issue 23: Move to A4
        phase_label = Text("Phase", font_size=24, color=COLOR_PHASE)
        self.place_at_grid(phase_label, 'A4')
        
        self.play(ReplacementTransform(freq_sine, phase_sine), Write(phase_label))
        self.play(phase_tracker.animate.set_value(PI), run_time=1.5)
        self.play(phase_tracker.animate.set_value(-PI), run_time=1.5)
        self.play(phase_tracker.animate.set_value(0), run_time=1)
        self.wait(1)
        self.play(FadeOut(phase_sine, phase_label, ax))

        # === Animation for Lecture Line 5 ===
        # Circular motion generates these smooth periodic oscillations.
        self.lecture[4].set_color(COLOR_CIRC)
        
        # Fixed positioning based on Issue 25: Circle at C2
        circle = Circle(radius=0.8, color=WHITE)
        self.place_at_grid(circle, 'C2')
        
        angle_tracker = ValueTracker(0)
        
        # Yellow dot tracing circular motion
        dot = Dot(color=COLOR_CIRC)
        dot.add_updater(lambda d: d.move_to(
            circle.get_center() + 
            np.array([
                circle.radius * np.cos(angle_tracker.get_value()), 
                circle.radius * np.sin(angle_tracker.get_value()), 
                0
            ])
        ))
        
        arm = always_redraw(lambda: Line(
            circle.get_center(), 
            dot.get_center(), 
            color=WHITE, 
            stroke_width=2
        ))
        
        # Fixed positioning based on Issue 25: wave_ax at C3-D6
        wave_ax = Axes(
            x_range=[0, 2*PI, PI/2], 
            y_range=[-1.2, 1.2, 1], 
            x_length=3.5, 
            y_length=2.0, 
            axis_config={"include_tip": False}
        ).set_stroke(opacity=0.3)
        self.place_in_area(wave_ax, 'C3', 'D6')
        
        # Drawing the sine wave as the dot rotates
        drawing_sine = always_redraw(lambda: wave_ax.plot(
            lambda x: np.sin(x), 
            x_range=[0, max(0.001, angle_tracker.get_value())], 
            color=WHITE
        ))
        
        # Connector line between circle dot and sine wave plot
        horizontal_line = always_redraw(lambda: Line(
            dot.get_center(), 
            wave_ax.c2p(angle_tracker.get_value(), np.sin(angle_tracker.get_value())), 
            color=GRAY, 
            stroke_opacity=0.5, 
            stroke_width=1
        ))
        
        self.play(Create(circle), Create(dot), Create(arm), Create(wave_ax))
        self.add(horizontal_line, drawing_sine)
        self.play(angle_tracker.animate.set_value(2*PI), run_time=6, rate_func=linear)
        self.wait(3)

# Update issues
# update_issue(23, under_review=True, resolution_note="Moved Amplitude, Frequency, and Phase labels from B4 to A4 to prevent overlap with the plot.")
# update_issue(24, under_review=True, resolution_note="Moved main Axes from B2-E6 to B3-E6 to provide more space from lecture notes.")
# update_issue(25, under_review=True, resolution_note="Moved circle to C2 and wave_ax to C3-D6 to utilize the top half of the grid and improve layout.")
