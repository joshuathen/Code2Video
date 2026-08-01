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
        self.setup_layout(
            "The Concept of Abstraction",
            [
                "Vectors are not just arrows; they are rule followers.",
                "A vector space is a collection of compatible objects.",
                "Adding or scaling these objects keeps them within the set."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Display a set of abstract shapes inside a container labeled 'Vector Space' (#FFFFFF).
        self.lecture[0].set_color(WHITE)
        
        container = RoundedRectangle(corner_radius=0.2, height=5.5, width=5.5, color=WHITE)
        self.place_in_area(container, 'A1', 'F6')
        
        container_label = Text("Vector Space", font_size=20, color=WHITE)
        # Resolved Issue 21: centering the label using place_in_area
        self.place_in_area(container_label, 'A3', 'A4', scale_factor=0.8)
        
        shape1 = Triangle(color=BLUE_E).scale(0.4)
        shape2 = Square(color=GREEN_E).scale(0.4)
        shape3 = Circle(color=RED_E).scale(0.4)
        shape4 = RegularPolygon(n=5, color=YELLOW_E).scale(0.4)
        
        self.place_at_grid(shape1, "B2")
        self.place_at_grid(shape2, "B5")
        self.place_at_grid(shape3, "E2")
        self.place_at_grid(shape4, "E5")
        
        self.play(
            Create(container),
            Write(container_label),
            FadeIn(shape1),
            FadeIn(shape2),
            FadeIn(shape3),
            FadeIn(shape4)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate two shapes merging (addition) to form a new shape within the container in light purple (#DDA0DD).
        self.lecture[1].set_color("#DDA0DD")
        
        addition_result = Star(n=5, color="#DDA0DD").scale(0.5)
        # Resolved Issue 22: centering the result using place_in_area
        self.place_in_area(addition_result, 'C3', 'D4', scale_factor=0.8)
        
        self.play(
            shape1.animate.move_to(addition_result.get_center()),
            shape2.animate.move_to(addition_result.get_center()),
            run_time=1.5
        )
        self.play(
            ReplacementTransform(VGroup(shape1, shape2), addition_result),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a shape being scaled (changing size) while remaining within the container in light orange (#FFDAB9).
        self.lecture[2].set_color("#FFDAB9")
        
        self.play(
            addition_result.animate.set_color("#FFDAB9"),
        )
        
        # Scale up
        self.play(
            addition_result.animate.scale(1.5),
            run_time=1
        )
        # Scale down
        self.play(
            addition_result.animate.scale(0.5),
            run_time=1
        )
        # Scale back to normal
        self.play(
            addition_result.animate.scale(1.33), 
            run_time=1
        )
        
        self.wait(2)
