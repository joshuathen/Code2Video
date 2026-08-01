from manim import *
import numpy as np

# Fix: Manim's config.get_dir() fails if the input file path contains curly braces
# like {iπ} because it attempts to .format() the path recursively. 
# We replace curly braces in the config paths to prevent the KeyError.
for _key in ["input_file", "output_dir", "media_dir"]:
    try:
        _val = str(config[_key])
        if "{" in _val or "}" in _val:
            config[_key] = _val.replace("{", "(").replace("}", ")")
    except:
        pass

# Fix for potential path formatting issues in environments with special characters in filenames
# Using Text instead of MathTex globally because the environment lacks a LaTeX distribution
MathTex = Text

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
        # Initializing the layout
        lecture_lines = [
            "Now, let's plug pi into our general formula.",
            "We start at the number 1 on the plane.",
            "Rotating by pi radians is exactly half a circle.",
            "This path leads us directly to the value -1.",
            "Thus, e raised to i times pi equals negative one."
        ]
        self.setup_layout("The Grand Reveal: Plugging in π", lecture_lines)

        # Pre-creating visual components
        axes = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-1.5, 1.5, 1],
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.8) # Issue 44 Fix
        
        unit_circle = Circle(radius=axes.x_axis.get_unit_size(), color=GRAY, stroke_opacity=0.5)
        unit_circle.move_to(axes.c2p(0, 0))

        angle_tracker = ValueTracker(0)
        
        vector = Arrow(
            axes.c2p(0, 0),
            axes.c2p(1, 0),
            buff=0,
            color=BLUE
        )
        
        def vector_updater(v):
            angle = angle_tracker.get_value()
            v.put_start_and_end_on(
                axes.c2p(0, 0),
                axes.c2p(np.cos(angle), np.sin(angle))
            )
        
        vector.add_updater(vector_updater)

        # Labels
        label_one = Text("1", font_size=24, color=BLUE)
        self.place_at_grid(label_one, 'D5', scale_factor=0.7) # Positioned near (1,0)
        
        label_neg_one = Text("-1", font_size=24, color=ORANGE)
        self.place_at_grid(label_neg_one, 'D2', scale_factor=0.5) # Issue 46 Fix

        # Formulas
        general_formula = MathTex("e^{ix} = cos(x) + i sin(x)", font_size=28, color=YELLOW)
        self.place_in_area(general_formula, 'A1', 'A6', scale_factor=1.0)

        simplification = MathTex("cos(pi) + i sin(pi) = -1 + 0", font_size=28, color=ORANGE)
        self.place_in_area(simplification, 'A1', 'A6', scale_factor=1.0)

        euler_formula = MathTex("e^{iπ} = -1", font_size=40, color="#FFD700")
        self.place_in_area(euler_formula, 'A1', 'A6', scale_factor=1.0) # Issue 45 Fix

        # === Animation for Lecture Line 1 ===
        # "Now, let's plug pi into our general formula."
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(axes), FadeIn(general_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We start at the number 1 on the plane."
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Create(unit_circle), FadeIn(vector), FadeIn(label_one))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Rotating by pi radians is exactly half a circle."
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        arc_path = Arc(radius=axes.x_axis.get_unit_size(), start_angle=0, angle=PI, color=GREEN)
        arc_path.move_to(axes.c2p(0, 0), aligned_edge=RIGHT).shift(RIGHT * axes.x_axis.get_unit_size())
        # Manual adjustment for arc to align with unit circle path
        arc_path.move_to(axes.c2p(0, 0.4), aligned_edge=DOWN).shift(DOWN*0.4) 
        # Simpler: just rotate the tracker
        self.play(angle_tracker.animate.set_value(PI), Create(arc_path), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This path leads us directly to the value -1."
        self.play(self.lecture[3].animate.set_color("#FF8C00"))
        self.play(FadeIn(label_neg_one))
        self.play(FadeOut(general_formula), FadeIn(simplification))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Thus, e raised to i times pi equals negative one."
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        self.play(FadeOut(simplification), FadeIn(euler_formula))
        self.play(euler_formula.animate.scale(1.2))
        self.wait(3)
