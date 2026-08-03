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
            "The Leap: What is 'Abstract'?",
            [
                "Modern math defines vectors by behavior, not appearance.",
                "Any set following specific rules forms a vector space.",
                "We move from \"what it is\" to \"how it behaves\"."
            ]
        )

        # Helper for grayscale grid
        def create_grid_image(intensities, size=0.4):
            squares = []
            for val in intensities:
                sq = Square(
                    side_length=size, 
                    fill_opacity=1, 
                    fill_color=interpolate_color(BLACK, WHITE, val), 
                    stroke_color=GREY_E, 
                    stroke_width=1
                )
                squares.append(sq)
            return VGroup(*squares).arrange_in_grid(rows=2, cols=2, buff=0)

        # === Animation for Lecture Line 1 ===
        # Morph a geometric arrow into a 2x2 grid representing a grayscale digital image. Label it 'Digital Vector'.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Resolution for Issue 26: Move arrow to B2, scale 0.6
        arrow = Arrow(start=LEFT, end=RIGHT, color=YELLOW)
        self.place_at_grid(arrow, "B2", scale_factor=0.6)
        
        grid_image = create_grid_image([0.2, 0.8, 0.5, 0.3])
        self.place_at_grid(grid_image, "B3", scale_factor=1.2)
        
        digital_label = Text("Digital Vector", font_size=20, color=YELLOW)
        self.place_at_grid(digital_label, "A3", scale_factor=0.8)
        
        self.play(Create(arrow))
        self.wait(1)
        self.play(ReplacementTransform(arrow, grid_image), Write(digital_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show two 2x2 grids (images) with different intensities. Add them to create a third grid with combined intensities.
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        grid1 = create_grid_image([0.1, 0.4, 0.2, 0.1])
        grid2 = create_grid_image([0.3, 0.2, 0.5, 0.6])
        grid_sum = create_grid_image([0.4, 0.6, 0.7, 0.7]) # Summed intensities
        
        plus = Text("+", font_size=30, color=WHITE)
        equals = Text("=", font_size=30, color=WHITE)
        
        # Resolution for Issue 27: Shift equation positions to D2-D6
        self.place_at_grid(grid1, "D2", scale_factor=1.0)
        self.place_at_grid(plus, "D3", scale_factor=1.0)
        self.place_at_grid(grid2, "D4", scale_factor=1.0)
        self.place_at_grid(equals, "D5", scale_factor=1.0)
        self.place_at_grid(grid_sum, "D6", scale_factor=1.0)
        
        self.play(
            FadeIn(grid1),
            Write(plus),
            FadeIn(grid2),
            Write(equals)
        )
        self.play(FadeIn(grid_sum))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display a set boundary (circle) containing both arrows and image grids. Label the set 'Vector Space' in light blue (#ADD8E6).
        self.play(self.lecture[2].animate.set_color("#ADD8E6"))
        
        # Area from E2 to F6 to avoid overlapping with lecture area (Col 1)
        set_boundary = Circle(color="#ADD8E6", stroke_width=4)
        self.place_in_area(set_boundary, "E2", "F6", scale_factor=2.2)
        
        # Resolution for Issue 28: Move set_label to F2
        set_label = Text("Vector Space", font_size=24, color="#ADD8E6")
        self.place_at_grid(set_label, "F2", scale_factor=0.8)
        
        # Elements inside the set
        small_arrow = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=YELLOW)
        small_grid = create_grid_image([0.4, 0.7, 0.2, 0.9], size=0.2)
        
        self.place_at_grid(small_arrow, "E4", scale_factor=0.5)
        self.place_at_grid(small_grid, "F5", scale_factor=0.5)
        
        self.play(Create(set_boundary), Write(set_label))
        self.play(
            FadeIn(small_arrow),
            FadeIn(small_grid)
        )
        self.wait(2)
