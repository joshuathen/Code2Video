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
        # Setup layout
        self.setup_layout("Visualizing Linear Combinations & Span", [
            "Every vector is a mix of fundamental basis elements.",
            "We call this \"mixing\" process a linear combination.",
            "Scaling and adding basis vectors creates a unique result.",
            "\"Span\" is the set of all possible combinations created.",
            "It represents the entire reach of your basis vectors."
        ])

        # ValueTrackers for amplitudes (weights)
        amp1 = ValueTracker(1.0)
        amp2 = ValueTracker(0.5)
        amp3 = ValueTracker(0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create basis waveforms (individual components)
        sine_wave = FunctionGraph(lambda x: np.sin(x * PI * 2), x_range=[-0.5, 0.5], color="#FF0000")
        square_wave = FunctionGraph(lambda x: np.sign(np.sin(x * PI * 2)), x_range=[-0.5, 0.5], color="#00FF00")
        sawtooth_wave = FunctionGraph(lambda x: 2 * (x - np.floor(x + 0.5)), x_range=[-0.5, 0.5], color="#0000FF")

        # Repositioning according to issues 32 and 33
        self.place_at_grid(sine_wave, 'A3', scale_factor=0.6)
        self.place_at_grid(square_wave, 'C3', scale_factor=0.6)
        self.place_at_grid(sawtooth_wave, 'E3', scale_factor=0.6)

        # Labels e1, e2, e3 (Issue 33)
        e1_label = MathTex("e_1", color="#FF0000", font_size=24)
        e2_label = MathTex("e_2", color="#00FF00", font_size=24)
        e3_label = MathTex("e_3", color="#0000FF", font_size=24)
        
        self.place_at_grid(e1_label, 'A2', scale_factor=0.8)
        self.place_at_grid(e2_label, 'C2', scale_factor=0.8)
        self.place_at_grid(e3_label, 'E2', scale_factor=0.8)

        self.play(
            Create(sine_wave), Create(square_wave), Create(sawtooth_wave),
            Write(e1_label), Write(e2_label), Write(e3_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Sliders: Horizontal line with a dot
        def create_slider(pos, color):
            line = Line(LEFT, RIGHT, color=GREY_A).scale(0.4)
            self.place_at_grid(line, pos)
            dot = Dot(color=color)
            dot.move_to(line.get_left())
            return VGroup(line, dot)

        slider1 = create_slider("A5", "#FF0000")
        slider2 = create_slider("C5", "#00FF00")
        slider3 = create_slider("E5", "#0000FF")

        self.play(Create(slider1), Create(slider2), Create(slider3))
        
        # Helper to map tracker value [0, 1] to slider position
        def get_slider_pos(slider, val):
            return slider[0].point_from_proportion(val)

        self.play(
            slider1[1].animate.move_to(get_slider_pos(slider1, 0.7)),
            slider2[1].animate.move_to(get_slider_pos(slider2, 0.4)),
            slider3[1].animate.move_to(get_slider_pos(slider3, 0.6)),
            amp1.animate.set_value(0.7),
            amp2.animate.set_value(0.4),
            amp3.animate.set_value(0.6),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Result wave (Linear Combination)
        # We'll use a sampled path to avoid complex functional graph updates
        result_center = (self.grid["C4"] + self.grid["F6"]) / 2
        
        # Initial empty wave
        combined_wave = VMobject(color="#FFFFFF", stroke_width=3)
        x_points = np.linspace(-1.5, 1.5, 100)
        
        def get_combined_points():
            a1 = amp1.get_value()
            a2 = amp2.get_value()
            a3 = amp3.get_value()
            pts = []
            for px in x_points:
                # Component functions
                v1 = a1 * np.sin(px * PI * 2)
                v2 = a2 * np.sign(np.sin(px * PI * 2))
                v3 = a3 * 2 * (px/2 - np.floor(px/2 + 0.5)) # Scale x for sawtooth to fit
                py = (v1 + v2 + v3) * 0.3 # Scale sum to fit
                pts.append(result_center + np.array([px, py, 0]))
            return pts

        combined_wave.set_points_as_corners(get_combined_points())
        
        # Add updater to the wave
        combined_wave.add_updater(lambda m: m.set_points_as_corners(get_combined_points()))
        
        result_label = Text("Result Sound", font_size=18, color=WHITE)
        self.place_at_grid(result_label, "D4")

        self.play(Create(combined_wave), Write(result_label))
        
        # Animate scaling to show "mixing"
        self.play(
            amp1.animate.set_value(0.2),
            amp2.animate.set_value(0.9),
            amp3.animate.set_value(0.3),
            slider1[1].animate.move_to(get_slider_pos(slider1, 0.2)),
            slider2[1].animate.move_to(get_slider_pos(slider2, 0.9)),
            slider3[1].animate.move_to(get_slider_pos(slider3, 0.3)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # "Span" gradient background
        # Create a rectangle covering the right side grid
        span_bg = Rectangle(
            width=6.0, height=5.5,
            fill_opacity=0.0,
            stroke_width=0
        )
        self.place_in_area(span_bg, "A1", "F6")
        
        # We'll use a color gradient fill
        span_bg.set_fill(
            color=[RED, GREEN, BLUE],
            opacity=0.2
        )
        
        span_text = Text("SPAN", font_size=40, color=WHITE, weight=BOLD).set_opacity(0.3)
        self.place_in_area(span_text, "A1", "F6")

        self.play(
            FadeIn(span_bg),
            Write(span_text),
            amp1.animate.set_value(0.5),
            amp2.animate.set_value(0.5),
            amp3.animate.set_value(0.5),
            slider1[1].animate.move_to(get_slider_pos(slider1, 0.5)),
            slider2[1].animate.move_to(get_slider_pos(slider2, 0.5)),
            slider3[1].animate.move_to(get_slider_pos(slider3, 0.5)),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Final sweep of parameters to show "entire reach"
        self.play(
            amp1.animate.set_value(1.0),
            amp2.animate.set_value(0.0),
            amp3.animate.set_value(1.0),
            slider1[1].animate.move_to(get_slider_pos(slider1, 1.0)),
            slider2[1].animate.move_to(get_slider_pos(slider2, 0.0)),
            slider3[1].animate.move_to(get_slider_pos(slider3, 1.0)),
            run_time=1.5
        )
        self.play(
            amp1.animate.set_value(0.0),
            amp2.animate.set_value(1.0),
            amp3.animate.set_value(0.0),
            slider1[1].animate.move_to(get_slider_pos(slider1, 0.0)),
            slider2[1].animate.move_to(get_slider_pos(slider2, 1.0)),
            slider3[1].animate.move_to(get_slider_pos(slider3, 0.0)),
            run_time=1.5
        )

        self.wait(2)
