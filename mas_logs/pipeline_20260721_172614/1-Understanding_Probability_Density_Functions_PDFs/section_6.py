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
        # Data from storyboard
        lecture_lines = [
            "Remember: for continuous variables, look at area, not height.",
            "The total area must always sum to exactly one.",
            "Use integrals to unlock probabilities in a continuous world."
        ]
        self.setup_layout("Summary and Key Takeaway", lecture_lines)
        
        # Colors
        COLOR_ORANGE = "#f39c12"
        COLOR_WHITE = "#ffffff"
        COLOR_PURPLE = "#9b59b6"
        COLOR_KITTEN = "#ecf0f1"
        COLOR_ROBOT = "#bdc3c7"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line in orange
        self.play(self.lecture[0].animate.set_color(COLOR_ORANGE))

        # Montage elements
        # Using SVG for kitten as requested in Issue 44
        try:
            kitten = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/kitten.svg")
            kitten.set_color(COLOR_KITTEN)
        except:
            # Fallback for local development or if file missing
            kitten = VGroup(
                Square(side_length=1.5, color=COLOR_KITTEN, fill_opacity=0.3),
                Text("Kitten", font_size=20, color=COLOR_KITTEN)
            )

        # Robot/Drone representation
        robot = VGroup(
            Square(side_length=1.0, color=COLOR_ROBOT, fill_opacity=0.5),
            Circle(radius=0.2, color=COLOR_ROBOT, fill_opacity=0.8).shift(UP*0.4),
            Text("Robot", font_size=20, color=COLOR_ROBOT).shift(DOWN*0.7)
        )
        
        integral_sym = MathTex(r"\int", font_size=80, color=COLOR_PURPLE)
        
        # Slogan text for Line 1
        slogan = Text("Area = Likelihood", font_size=36, color=COLOR_ORANGE)
        
        # Positions based on Issues 41 and 42
        self.place_at_grid(kitten, "B3", scale_factor=0.6)        # Issue 41: B3
        self.place_at_grid(integral_sym, "B4", scale_factor=1.0) # Issue 41: B4
        self.place_at_grid(robot, "B5", scale_factor=0.6)        # Balanced row
        self.place_in_area(slogan, "C3", "D5", scale_factor=1.0) # Issue 42: C3-D5

        # Entrance animations for the montage
        self.play(FadeIn(kitten), run_time=0.4)
        self.play(FadeIn(integral_sym), run_time=0.4)
        self.play(FadeIn(robot), run_time=0.4)
        self.play(Write(slogan), run_time=1.0)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight (1 to 2)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_WHITE)
        )

        # Gaussian curve representation to illustrate "Total Area = 1"
        axes = Axes(
            x_range=[-3, 3],
            y_range=[0, 0.5],
            axis_config={"include_tip": False},
            x_length=3.5,
            y_length=1.5
        )
        
        def normal_pdf(x):
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        
        curve = axes.plot(normal_pdf, color=COLOR_WHITE)
        area_obj = axes.get_area(curve, x_range=[-3, 3], color=COLOR_WHITE, opacity=0.3)
        label_one = MathTex(r"\text{Total Area} = 1", color=COLOR_WHITE, font_size=32)
        
        # Group and position the curve visualization
        curve_group = VGroup(axes, curve, area_obj, label_one)
        label_one.next_to(curve, UP, buff=0.1)
        # Issue 43: Center the curve group
        self.place_in_area(curve_group, "E3", "F5", scale_factor=0.8)

        # Visualization sequence
        self.play(Create(axes), Create(curve), run_time=1.2)
        self.play(FadeIn(area_obj), Write(label_one))
        
        # Highlight effect: subtle stroke pulse
        self.play(curve.animate.set_stroke(width=8), run_time=0.5)
        self.play(curve.animate.set_stroke(width=2), run_time=0.5)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight (2 to 3)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_PURPLE)
        )
        
        # Emphasis on Integral Symbol as a closing beat
        self.play(
            integral_sym.animate.scale(1.4).set_color(WHITE),
            rate_func=there_and_back,
            run_time=2.0
        )
        
        self.wait(3.5)
