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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Million-Dollar Mystery: The Riemann Hypothesis"
        lines = [
            "We focus on the critical strip in the complex plane.",
            "Trivial zeros appear at negative even integers.",
            "We draw the critical line at real part one-half.",
            "Riemann hypothesized all non-trivial zeros lie here.",
            "This vertical line remains mathematics' greatest unsolved mystery."
        ]
        self.setup_layout(title, lines)
        
        # Colors for lecture lines matching the visual elements
        colors = [BLUE_B, WHITE, RED, "#FFD700", YELLOW]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Setup Complex Plane (NumberPlane used here)
        axes = NumberPlane(
            x_range=[-8, 4, 2],
            y_range=[-4, 4, 2],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        ).add_coordinates(label_constructor=Text, font_size=16)
        
        # Fix for Issue 48: Position axes in specified area
        self.place_in_area(axes, 'B1', 'F6', scale_factor=0.65)
        
        # Formula for Zeta
        zeta_formula = Text("ζ(s) = Σ 1/nˢ", font_size=32, color=BLUE_B)
        # Fix for Issue 47: Position formula at A1
        self.place_at_grid(zeta_formula, 'A1', scale_factor=0.8)
        
        # Highlight Critical Strip (0 < Re(s) < 1)
        strip = Rectangle(
            width=axes.x_axis.get_unit_size() * 1,
            height=axes.y_axis.get_unit_size() * 8,
            fill_color=BLUE_E,
            fill_opacity=0.3,
            stroke_width=0
        )
        strip.move_to(axes.c2p(0.5, 0))

        self.play(Create(axes), Write(zeta_formula))
        self.play(FadeIn(strip))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Mark trivial zeros at s = -2, -4, -6 with white X marks
        trivial_points = [-2, -4, -6]
        trivial_zeros = VGroup(*[
            Text("X", color=WHITE, font_size=24).move_to(axes.c2p(p, 0))
            for p in trivial_points
        ])
        
        self.play(Create(trivial_zeros))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Draw bold red line at Re(s) = 0.5 (The Critical Line)
        critical_line = Line(
            start=axes.c2p(0.5, -4),
            end=axes.c2p(0.5, 4),
            color="#FF0000",
            stroke_width=6
        )
        
        # Label for Critical Line
        line_label = Text("Re(s) = 1/2", color="#FF0000", font_size=24)
        # Fix for Issue 49: Position label at B5
        self.place_at_grid(line_label, 'B5', scale_factor=0.7)
        
        self.play(Create(critical_line))
        self.play(Write(line_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # Gold dots representing non-trivial zeros on the critical line
        # Representative y-values for demonstration
        zero_y_values = [1.4, 2.1, -1.4, -2.1, 0.5, -0.5]
        non_trivial_zeros = VGroup(*[
            Dot(axes.c2p(0.5, y), color="#FFD700", radius=0.08)
            for y in zero_y_values
        ])
        
        self.play(LaggedStartMap(FadeIn, non_trivial_zeros, lag_ratio=0.2, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        # Zoom out effect: scale down the visualization group
        viz_group = VGroup(axes, strip, trivial_zeros, critical_line, non_trivial_zeros)
        
        self.play(
            viz_group.animate.scale(0.5).move_to(self.grid["D3"]),
            line_label.animate.scale(0.8).next_to(self.grid["D3"], UP, buff=0.2),
            run_time=2
        )
        self.wait(2)
