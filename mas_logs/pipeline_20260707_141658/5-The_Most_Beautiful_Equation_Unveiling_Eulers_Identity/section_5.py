from manim import *

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
        self.setup_layout("Defining the Formula: Euler's Path", [
            "Euler's formula describes movement around a circle.",
            "The exponent x represents the angle in radians.",
            "Cosine tracks the horizontal position on the circle.",
            "Sine tracks the vertical position on the imaginary axis.",
            "Together, they map a point on the complex plane."
        ])
        
        # Colors
        GRAY = "#808080"
        YELLOW = "#FFFFE0"
        BLUE = "#ADD8E6"
        GREEN = "#90EE90"
        WHITE_COLOR = "#FFFFFF"

        # Initialize all lecture text to gray
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE_COLOR)
        
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4.0,
            y_length=4.0,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, "B2", "E5")
        origin = axes.get_center()
        
        circle = Circle(radius=axes.x_axis.unit_size, color=GRAY)
        circle.move_to(origin)
        
        # Replaced MathTex with Text to resolve FileNotFoundError: 'latex'
        exp_label = Text("e^ix", color=GRAY, font_size=32)
        self.place_at_grid(exp_label, 'B5', scale_factor=0.8)

        self.play(Create(axes), Create(circle), Write(exp_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(WHITE_COLOR)
        
        angle_val = PI / 3
        point_on_circle = axes.c2p(np.cos(angle_val), np.sin(angle_val))
        dot = Dot(color=YELLOW).move_to(point_on_circle)
        radius_line = Line(origin, point_on_circle, color=YELLOW)
        arc = Arc(radius=0.5, start_angle=0, angle=angle_val, arc_center=origin, color=YELLOW)
        # Replaced MathTex with Text
        label_x = Text("x", color=YELLOW, font_size=28).next_to(arc, RIGHT, buff=0.1)

        self.play(Create(radius_line), Create(arc), Write(label_x), FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(WHITE_COLOR)
        
        cos_line = Line(
            axes.c2p(0, 0), 
            axes.c2p(np.cos(angle_val), 0), 
            color=BLUE, 
            stroke_width=6
        )
        # Replaced MathTex with Text
        label_cos = Text("cos(x)", color=BLUE, font_size=28).next_to(cos_line, DOWN, buff=0.1)
        
        self.play(Create(cos_line), Write(label_cos))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(WHITE_COLOR)
        
        sin_line = Line(
            axes.c2p(np.cos(angle_val), 0), 
            axes.c2p(np.cos(angle_val), np.sin(angle_val)), 
            color=GREEN, 
            stroke_width=6
        )
        # Replaced MathTex with Text
        sin_label = Text("i sin(x)", color=GREEN, font_size=28)
        self.place_in_area(sin_label, 'C5', 'D5', scale_factor=0.6)
        
        self.play(Create(sin_line), Write(sin_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(WHITE_COLOR)
        
        # Replaced MathTex with Text
        euler_formula = Text("e^ix = cos(x) + i sin(x)", font_size=36, color=WHITE_COLOR)
        self.place_in_area(euler_formula, 'F1', 'F6', scale_factor=0.8)
        
        self.play(Write(euler_formula))
        self.wait(2)
