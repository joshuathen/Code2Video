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
        # Initializing the layout
        self.setup_layout(
            "The Prior: Setting the Stage", 
            [
                'Let this unit square represent all possible outcomes.', 
                'In our forest, ten percent are rare Golden Squirrels.', 
                'This vertical strip shows our initial prior belief.', 
                'The other ninety percent are common Grey Squirrels.', 
                'Area represents probability in this geometric model.'
            ]
        )
        
        # Colors
        COLOR_GOLD = "#FFD700"
        COLOR_GREY = "#808080"
        COLOR_WHITE = "#FFFFFF"

        # Constructing the geometric components
        gold_rect = Rectangle(width=0.4, height=4, fill_color=COLOR_GOLD, fill_opacity=0.8, stroke_width=0)
        grey_rect = Rectangle(width=3.6, height=4, fill_color=COLOR_GREY, fill_opacity=0.8, stroke_width=0)
        strips = VGroup(gold_rect, grey_rect).arrange(RIGHT, buff=0)
        
        square_outline = Square(side_length=4, color=COLOR_WHITE, stroke_width=2)
        divider = Line(UP*2, DOWN*2, color=COLOR_WHITE, stroke_width=2)
        
        # Labels - Using Text instead of MathTex to avoid LaTeX environment errors
        label_sample = Text("Sample Space", color=COLOR_WHITE, font_size=24)
        label_gold = Text("P(Gold)=0.1", color=COLOR_GOLD, font_size=24)
        label_grey = Text("P(Grey)=0.9", color=COLOR_GREY, font_size=32)

        # Positioning (using area and grid)
        diagram_group = VGroup(strips, square_outline, divider)
        self.place_in_area(diagram_group, "A1", "F6")
        
        # Placing labels relative to the grid
        self.place_at_grid(label_sample, "F3", scale_factor=1.0)
        self.place_at_grid(label_gold, "B2", scale_factor=0.7) 
        self.place_at_grid(label_grey, "B4", scale_factor=1.0) 

        # Adjust divider line to its 10% position within the square
        divider.move_to(square_outline.get_center() + LEFT * 1.6)

        # === Animation for Lecture Line 1 ===
        self.play(
            self.lecture[0].animate.set_color(COLOR_WHITE),
            Create(square_outline),
            Write(label_sample)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[1].animate.set_color(COLOR_GOLD),
            Create(divider),
            FadeIn(gold_rect)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[2].animate.set_color(COLOR_GOLD),
            Write(label_gold)
        )
        self.play(
            Indicate(gold_rect, color=COLOR_WHITE, scale_factor=1.05),
            Flash(gold_rect, color=COLOR_WHITE, line_length=0.2, flash_radius=0.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[3].animate.set_color(COLOR_GREY),
            FadeIn(grey_rect),
            Write(label_grey)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[4].animate.set_color(COLOR_WHITE),
            label_gold.animate.scale(1.2),
            label_grey.animate.scale(1.1)
        )
        self.play(
            label_gold.animate.scale(1/1.2),
            label_grey.animate.scale(1/1.1)
        )
        self.wait(2)