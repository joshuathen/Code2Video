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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching content from shared state
        title = "Scalar vs. Vector: More than just a Number"
        lines = [
            "Some values only describe size, called scalars.",
            "Speed, like 100 km/h, is a common scalar.",
            "Vectors add direction to the description of size.",
            "Velocity is a vector: 100 km/h towards north.",
            "We represent vectors as arrows with specific lengths."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        SCALAR_COLOR = "#00FF00"
        VECTOR_COLOR = "#0000FF"
        HIGHLIGHT_COLOR = "#FFFFFF"

        # Mobjects
        scalar_label = Text("SCALAR", color=SCALAR_COLOR)
        scalar_val = Text("100 km/h", color=WHITE)
        vector_label = Text("VECTOR", color=VECTOR_COLOR)
        # Using a long arrow to represent the vector
        vector_arrow = Arrow(start=LEFT, end=RIGHT, color=VECTOR_COLOR, buff=0)
        
        # Positioning using the 6x6 grid system
        # Resolving Issue 21: Position scalar_label at B4
        self.place_at_grid(scalar_label, 'B4', scale_factor=0.8)
        # Position scalar_val at C4 to align with its label
        self.place_at_grid(scalar_val, 'C4', scale_factor=0.7)
        
        # Resolving Issue 22: Position vector_label at E4
        self.place_at_grid(vector_label, 'E4', scale_factor=0.8)
        # Resolving Issue 23: Position vector_arrow in area F3-F5
        self.place_in_area(vector_arrow, 'F3', 'F5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "Some values only describe size, called scalars."
        self.play(self.lecture[0].animate.set_color(SCALAR_COLOR))
        self.play(FadeIn(scalar_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Speed, like 100 km/h, is a common scalar."
        self.play(self.lecture[1].animate.set_color(SCALAR_COLOR))
        self.play(Write(scalar_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Vectors add direction to the description of size."
        self.play(self.lecture[2].animate.set_color(VECTOR_COLOR))
        # Initial vector representation: just the arrow
        self.play(GrowArrow(vector_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Velocity is a vector: 100 km/h towards north."
        self.play(self.lecture[3].animate.set_color(VECTOR_COLOR))
        self.play(FadeIn(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We represent vectors as arrows with specific lengths."
        self.play(self.lecture[4].animate.set_color(HIGHLIGHT_COLOR))
        # Highlight magnitude of the arrow using Indicate as per L004
        self.play(Indicate(vector_arrow, color=HIGHLIGHT_COLOR))
        self.wait(2)
