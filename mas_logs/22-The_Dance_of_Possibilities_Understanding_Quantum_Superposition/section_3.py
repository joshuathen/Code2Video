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
        # Initialize title and lecture lines based on the script
        self.setup_layout(
            "Visualizing the State Vector", 
            [
                'We visualize these states using a geometric state vector.', 
                'This arrow points up to represent the "Zero" state.', 
                'It rotates right to represent the "One" state.', 
                'Diagonally, the arrow represents a mix of both states.', 
                'Its direction shows the weight of each classical possibility.'
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
        # Fixed positioning to resolve Issues 32, 33, and 34
        self.place_in_area(geometry_group, 'B3', 'E5', scale_factor=0.7)
        
        center = circle.get_center()
        # Actual radius after scaling
        current_radius = circle.width / 2
        
        self.play(Create(circle), Create(axis_v), Create(axis_h), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This arrow points up to represent the "Zero" state.
        # Draw a vector arrow from the center to the top edge (label '|0⟩') in #FFFFFF.
        self.lecture[1].set_color(WHITE_COLOR)
        
        vector = Arrow(center, center + UP * current_radius, buff=0, color=WHITE_COLOR, stroke_width=4)
        label_0 = Text("|0⟩", color=WHITE_COLOR, font_size=32)
        label_0.next_to(center + UP * current_radius, UP, buff=0.2)
        
        self.play(GrowArrow(vector), Write(label_0))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It rotates right to represent the "One" state.
        # Rotate the vector arrow to point to the right edge (label '|1⟩') in #FFFFFF.
        self.lecture[2].set_color(WHITE_COLOR)
        
        label_1 = Text("|1⟩", color=WHITE_COLOR, font_size=32)
        label_1.next_to(center + RIGHT * current_radius, RIGHT, buff=0.2)

        self.play(
            Rotate(vector, angle=-PI/2, about_point=center),
            ReplacementTransform(label_0, label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Diagonally, the arrow represents a mix of both states.
        # Rotate the vector arrow to a 45-degree angle and change its color to #FFD700.
        self.lecture[3].set_color(GOLD_COLOR)
        
        # Current state: pointing RIGHT. Rotate counter-clockwise by PI/4.
        self.play(
            Rotate(vector, angle=PI/4, about_point=center),
            vector.animate.set_color(GOLD_COLOR),
            FadeOut(label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Its direction shows the weight of each classical possibility.
        # Label the diagonal vector '|ψ⟩' in #FFD700 to indicate the superposition state.
        self.lecture[4].set_color(GOLD_COLOR)
        
        label_psi = Text("|ψ⟩", color=GOLD_COLOR, font_size=32)
        # Position label near the tip of the diagonal vector
        tip_pos = center + (RIGHT * np.cos(PI/4) + UP * np.sin(PI/4)) * current_radius
        label_psi.next_to(tip_pos, UR, buff=0.1)

        self.play(Write(label_psi))
        self.wait(2)
