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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        self.setup_layout(
            "Visualizing the State Vector", 
            [
                'We visualize quantum states using a simple geometric arrow.', 
                'Vertical represents "Heads", while horizontal represents "Tails".', 
                'A classical coin points strictly along one axis.', 
                'The quantum arrow points diagonally, representing a mix.', 
                'This direction shows the weight of each possibility.'
            ]
        )
        
        # Constants
        RADIUS = 1.8
        WHITE_COLOR = "#FFFFFF"
        GOLD_COLOR = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Draw a white circle (#FFFFFF) with a vertical line (up) and a horizontal line (right).
        self.lecture[0].set_color(WHITE_COLOR)
        
        circle = Circle(radius=RADIUS, color=WHITE_COLOR)
        # Vertical and horizontal axes
        axis_v = Line(ORIGIN, UP * RADIUS, color=WHITE_COLOR)
        axis_h = Line(ORIGIN, RIGHT * RADIUS, color=WHITE_COLOR)
        
        geometry_group = VGroup(circle, axis_v, axis_h)
        self.place_in_area(geometry_group, 'A3', 'E6')
        
        # Coordinate system center derived from circle position
        center = circle.get_center()
        
        self.play(Create(circle), Create(axis_v), Create(axis_h), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vertical represents "Heads", while horizontal represents "Tails".
        # Draw a vector arrow from the center to the top edge (label '|0⟩') in #FFFFFF.
        self.lecture[1].set_color(WHITE_COLOR)
        
        # Vector pointing UP (|0>)
        vector = Arrow(center, center + UP * RADIUS, buff=0, color=WHITE_COLOR, stroke_width=4)
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_0 = Text("|0⟩", color=WHITE_COLOR, font_size=36)
        label_0.next_to(center + UP * RADIUS, UP, buff=0.2)
        
        heads_label = Text("Heads", font_size=20, color=WHITE_COLOR)
        heads_label.next_to(center + UP * RADIUS, LEFT, buff=0.2)
        
        tails_label = Text("Tails", font_size=20, color=WHITE_COLOR)
        tails_label.next_to(center + RIGHT * RADIUS, DOWN, buff=0.2)

        self.play(GrowArrow(vector), Write(label_0), FadeIn(heads_label), FadeIn(tails_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A classical coin points strictly along one axis.
        # Rotate the vector arrow to point to the right edge (label '|1⟩') in #FFFFFF.
        self.lecture[2].set_color(WHITE_COLOR)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_1 = Text("|1⟩", color=WHITE_COLOR, font_size=36)
        label_1.next_to(center + RIGHT * RADIUS, RIGHT, buff=0.2)

        self.play(
            Rotate(vector, angle=-PI/2, about_point=center),
            ReplacementTransform(label_0, label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The quantum arrow points diagonally, representing a mix.
        # Rotate the vector arrow to a 45-degree angle and change its color to #FFD700.
        self.lecture[3].set_color(GOLD_COLOR)
        
        # Current state: pointing RIGHT. Rotate counter-clockwise by PI/4 to get 45 deg between UP and RIGHT.
        self.play(
            Rotate(vector, angle=PI/4, about_point=center),
            vector.animate.set_color(GOLD_COLOR),
            FadeOut(label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This direction shows the weight of each possibility.
        # Label the diagonal vector '|ψ⟩' in #FFD700 to indicate the superposition state.
        self.lecture[4].set_color(GOLD_COLOR)
        
        # Replaced MathTex with Text to avoid LaTeX dependency error
        label_psi = Text("|ψ⟩", color=GOLD_COLOR, font_size=36)
        # Position label_psi near the tip of the diagonal vector
        tip_pos = center + (RIGHT * np.cos(PI/4) + UP * np.sin(PI/4)) * RADIUS
        label_psi.next_to(tip_pos, UR, buff=0.1)

        self.play(Write(label_psi))
        self.wait(2)