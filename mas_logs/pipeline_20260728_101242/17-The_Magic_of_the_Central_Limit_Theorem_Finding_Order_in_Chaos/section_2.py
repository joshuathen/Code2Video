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
        # COLORS
        ORANGE_COLOR = "#FFA500"
        MAGENTA_COLOR = "#FF00FF"
        CYAN_COLOR = "#00FFFF"

        self.setup_layout(
            "Prerequisite: The Building Blocks",
            [
                "Every population has a mean and standard deviation.",
                "A distribution shows how often different values occur.",
                "Our \"Chaos Island\" monsters have a flat distribution."
            ]
        )

        # Pre-create elements
        # Axes - Fix Issue 25: Move to C2-F6 to avoid crowding
        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 1, 0.5],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": False, "font_size": 18, "color": GRAY}
        )
        self.place_in_area(axes, "C2", "F6")

        # Mean Line and Label
        mu_val = 50
        mu_line = Line(
            axes.c2p(mu_val, 0),
            axes.c2p(mu_val, 0.8),
            color=ORANGE_COLOR
        )
        mu_label = MathTex(r"\mu", color=ORANGE_COLOR, font_size=36)
        # Fix Issue 26: Positioning the mean label with a grid anchor
        self.place_at_grid(mu_label, "B4", scale_factor=0.6)

        # Standard Deviation Arrow and Label
        # Fix Issue 27: Applying grid-based scaling to standard deviation arrow
        sigma_arrow = DoubleArrow(
            color=MAGENTA_COLOR,
            buff=0,
            tip_length=0.15
        )
        self.place_in_area(sigma_arrow, "D3", "D5", scale_factor=0.8)
        sigma_label = MathTex(r"\sigma", color=MAGENTA_COLOR, font_size=36)
        sigma_label.next_to(sigma_arrow, DOWN, buff=0.1)

        # Asset Integration - Issue 20
        # Load the monster icon SVG as specified in the storyboard
        monsters_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monsters.svg")
        monsters_icon.set_color(CYAN_COLOR)
        self.place_at_grid(monsters_icon, "B2", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Highlight first line and show statistical parameters
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        self.play(Create(axes), FadeIn(monsters_icon))
        self.play(Create(mu_line), Write(mu_label))
        self.play(Create(sigma_arrow), Write(sigma_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition to the concept of a distribution
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CYAN_COLOR)
        )

        # Representation of the uniform distribution (flat rectangular graph)
        uniform_rect = Rectangle(
            width=axes.x_axis.get_length(),
            height=axes.y_axis.get_unit_size() * 0.5,
            fill_color=CYAN_COLOR,
            fill_opacity=0.3,
            stroke_color=CYAN_COLOR
        )
        # Center the rectangle on the axes for a uniform appearance
        uniform_rect.move_to(axes.c2p(50, 0.25))

        self.play(FadeIn(uniform_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Emphasize the "flat" nature of the distribution on Chaos Island
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(CYAN_COLOR)
        )

        flat_top = Line(
            axes.c2p(0, 0.5),
            axes.c2p(100, 0.5),
            color=CYAN_COLOR,
            stroke_width=6
        )

        self.play(Create(flat_top))
        self.play(Flash(flat_top, color=CYAN_COLOR, line_length=0.3))
        self.wait(2)

        # Cleanup for next section transition
        self.play(
            FadeOut(flat_top), 
            FadeOut(uniform_rect), 
            FadeOut(axes), 
            FadeOut(mu_line), 
            FadeOut(mu_label), 
            FadeOut(sigma_arrow), 
            FadeOut(sigma_label),
            FadeOut(monsters_icon)
        )
        self.wait(1)
