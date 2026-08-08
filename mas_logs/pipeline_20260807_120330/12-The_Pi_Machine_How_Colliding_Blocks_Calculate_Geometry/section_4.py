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
        # Data from storyboard
        title = "The Circle Transformation"
        lecture_lines = [
            "Energy conservation defines a horizontal ellipse in velocity space.",
            "We rescale the vertical axis using the mass ratio.",
            "This transformation stretches the ellipse into a perfect circle.",
            "Each collision moves our state to a new point.",
            "This movement creates a specific angle from the center."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_ELLIPSE = "#FF00FF" # Magenta
        COLOR_CIRCLE = "#00FFFF"  # Cyan
        COLOR_POINT = "#FFFF00"   # Yellow
        COLOR_RADIUS = "#FFFFFF"  # White
        COLOR_LABEL_TEXT = "#FFFFFF" # White
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # "Energy conservation defines a horizontal ellipse in velocity space."
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Axes centered at D4
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_at_grid(axes, "D4")
        
        # horizontal ellipse: x^2/1^2 + y^2/0.25^2 = 1
        # width 2.0, height 0.5
        ellipse = Ellipse(width=2.0, height=0.5, color=COLOR_ELLIPSE)
        ellipse.move_to(axes.c2p(0, 0))
        
        label_v = MathTex("v", color=WHITE)
        label_V = MathTex("V", color=WHITE)
        
        # Issue 32 Fix: place_at_grid(label_V, 'B3', scale_factor=0.8)
        self.place_at_grid(label_v, "D6", scale_factor=0.8)
        self.place_at_grid(label_V, "B3", scale_factor=0.8)
        
        self.play(Create(axes), Write(label_v), Write(label_V))
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We rescale the vertical axis using the mass ratio."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Transformation label
        # Issue 31 Fix: place_in_area(label_V_transformed, 'B3', 'B5', scale_factor=0.8)
        label_V_transformed = MathTex(r"V \to V \sqrt{M/m}", color=COLOR_LABEL_TEXT)
        self.place_in_area(label_V_transformed, "B3", "B5", scale_factor=0.8)
        
        self.play(FadeIn(label_V_transformed))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This transformation stretches the ellipse into a perfect circle."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Circle with radius 1.0
        circle = Circle(radius=1.0, color=COLOR_CIRCLE)
        circle.move_to(axes.c2p(0, 0))
        
        # Morphing ellipse to circle
        self.play(
            ReplacementTransform(ellipse, circle),
            FadeOut(label_V_transformed),
            FadeOut(label_V)
        )
        
        # New label for the transformed axis
        new_label_V = MathTex(r"V'", color=WHITE)
        self.place_at_grid(new_label_V, "B3", scale_factor=0.8)
        self.play(FadeIn(new_label_V))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Each collision moves our state to a new point."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Start point at D5 (1 unit right of center D4)
        # End point at C4 (1 unit up from center D4)
        start_pt = axes.c2p(1, 0)
        end_pt = axes.c2p(0, 1)
        
        dot = Dot(start_pt, color=COLOR_POINT)
        self.play(FadeIn(dot))
        
        # Arc path from 0 to 90 degrees
        arc_path = Arc(radius=1.0, start_angle=0, angle=PI/2, arc_center=axes.c2p(0,0))
        
        # Animate jumping to C4 along the arc
        self.play(MoveAlongPath(dot, arc_path), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This movement creates a specific angle from the center."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Line from center to dot (which is now at C4)
        center_pt = axes.c2p(0, 0)
        radius_line = Line(center_pt, dot.get_center(), color=COLOR_RADIUS)
        
        # Angle arc and label theta
        # Using place_in_area for theta to avoid manual positioning
        angle_arc = Arc(radius=0.3, start_angle=0, angle=PI/2, arc_center=center_pt, color=WHITE)
        label_theta = MathTex(r"\theta", color=WHITE)
        self.place_in_area(label_theta, "C4", "D5", scale_factor=0.6)
        
        # Issue 30 Fix: label_R (R = sqrt(2E)) at F3-F4
        label_R = MathTex(r"R = \sqrt{2E}", color=COLOR_RADIUS)
        self.place_in_area(label_R, "F3", "F4", scale_factor=0.7)
        
        self.play(Create(radius_line))
        self.play(Create(angle_arc), Write(label_theta))
        self.play(Write(label_R))
        self.wait(2)
        
        # Final cleanup for consistency
        self.lecture[4].set_color(WHITE)
        self.wait(2)
