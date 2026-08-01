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

# The ImportError indicates that 'Section2Scene' could not be found
# in 'test_section_2.py'. This means the class 'Section2Scene' was
# either not defined, misspelled, or commented out in that file.
# We will define a placeholder for 'Section2Scene' here, assuming
# it is intended to be a Manim scene, possibly inheriting from TeachingScene.
class Section2Scene(TeachingScene):
    def construct(self):
        # Example usage of setup_layout from TeachingScene
        self.setup_layout(
            title_text="Introduction to Section 2",
            lecture_lines=[
                "- Point 1 for Section 2",
                "- Point 2 for Section 2",
                "- Point 3 for Section 2"
            ]
        )
        self.play(FadeIn(self.title, shift=UP), FadeIn(self.lecture, shift=LEFT))
        self.wait(2)

        # Add specific animations for Section 2 here
        example_circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        self.place_at_grid(example_circle, "C3")
        self.play(Create(example_circle))
        self.wait(1)

        example_square = Square(side_length=1, color=RED, fill_opacity=0.5)
        self.place_at_grid(example_square, "D4")
        self.play(Transform(example_circle, example_square))
        self.wait(2)
        self.play(FadeOut(example_circle), FadeOut(self.title), FadeOut(self.lecture))
        self.wait(1)
