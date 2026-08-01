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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup content
        title = "The Grand Destination: e^iπ = -1"
        lines = [
            "We start at one on the real axis.",
            "The i exponent turns our path into a circle.",
            "We travel a distance of exactly pi.",
            "This journey lands us perfectly at negative one.",
            "Add one, and we arrive back at zero."
        ]
        self.setup_layout(title, lines)

        # Initialize visual elements
        axes = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLUE_D},
            background_line_style={"stroke_opacity": 0.4}
        )
        # Resolved Issue 50: Lower axes position to avoid overlap
        self.place_in_area(axes, "C2", "F5")
        
        # Real/Imaginary Labels (using Text to avoid LaTeX requirement)
        labels = axes.get_axis_labels(
            x_label=Text("Re", font_size=20), 
            y_label=Text("Im", font_size=20)
        )
        
        # Dot at (1, 0)
        dot = Dot(axes.c2p(1, 0), color=YELLOW, radius=0.1)
        
        # Formula elements (using Text to avoid LaTeX requirement)
        formula_1 = Text("e^iπ = -1", font_size=48, color="#FFFFFF")
        # Resolved Issue 51: Center formula horizontally
        self.place_in_area(formula_1, "A3", "A5")
        
        formula_final = Text("e^iπ + 1 = 0", font_size=48, color="#FFFFFF")
        # Resolved Issue 52: Center final formula horizontally
        self.place_in_area(formula_final, "A3", "A5")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Write(labels))
        self.play(FadeIn(dot, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Show path outline
        arc_path = Arc(radius=axes.get_x_unit_size(), start_angle=0, angle=PI, color=WHITE, stroke_width=2)
        arc_path.move_to(axes.c2p(0,0))
        self.play(Create(arc_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Track rotation
        angle_tracker = ValueTracker(0)
        dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(angle_tracker.get_value()), np.sin(angle_tracker.get_value()))))
        
        self.play(angle_tracker.animate.set_value(PI), run_time=3, rate_func=linear)
        dot.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Landing pulse effect
        pulse = Circle(radius=0.1, color=YELLOW, stroke_width=4).move_to(axes.c2p(-1, 0))
        self.add(pulse)
        self.play(
            pulse.animate.scale(5).set_stroke(opacity=0),
            FadeIn(formula_1, shift=UP),
            run_time=1.5
        )
        self.remove(pulse)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        # Transform formula
        self.play(TransformMatchingShapes(formula_1, formula_final))
        self.play(formula_final.animate.scale(1.2).set_color(YELLOW), run_time=1)
        self.wait(3)
