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
        # Initializing the layout with mandatory script lines
        self.setup_layout(
            "The Prior: Setting the Stage", 
            [
                'This unit square represents our entire sample space.', 
                'We divide it to represent different possible outcomes.', 
                'Ten percent represent rare Golden Squirrels, others are Grey.', 
                'These areas show the initial probability of each group.', 
                'This gold strip is our prior belief for Golden Squirrels.'
            ]
        )
        
        # Colors
        COLOR_GOLD = "#FFD700"
        COLOR_GREY = "#808080"
        COLOR_WHITE = "#FFFFFF"

        # Constructing the geometric components
        # Square side 4. 10% of 4 is 0.4.
        gold_rect = Rectangle(width=0.4, height=4, fill_color=COLOR_GOLD, fill_opacity=0.8, stroke_width=0)
        grey_rect = Rectangle(width=3.6, height=4, fill_color=COLOR_GREY, fill_opacity=0.8, stroke_width=0)
        strips = VGroup(gold_rect, grey_rect).arrange(RIGHT, buff=0)
        
        square_outline = Square(side_length=4, color=COLOR_WHITE, stroke_width=2)
        divider = Line(UP*2, DOWN*2, color=COLOR_WHITE, stroke_width=2)
        
        # Labels
        label_sample = Text("Sample Space", color=COLOR_WHITE, font_size=24)
        label_gold = Text("P(Gold)=0.1", color=COLOR_GOLD, font_size=24)
        label_grey = Text("P(Grey)=0.9", color=COLOR_GREY, font_size=32)

        # Positioning (using area and grid)
        diagram_group = VGroup(strips, square_outline, divider)
        # ISSUE 24: Moved from A1-F6 to A1-E6 to prevent label_sample overlap
        self.place_in_area(diagram_group, "A1", "E6")
        
        # Adjust divider line to its 10% position within the square
        divider.move_to(square_outline.get_center() + LEFT * 1.6)

        # ISSUE 22: Moved label_gold to B1 and scaled to 0.6
        # ISSUE 23: Moved label_grey to D4
        self.place_at_grid(label_sample, "F3", scale_factor=1.0)
        self.place_at_grid(label_gold, "B1", scale_factor=0.6) 
        self.place_at_grid(label_grey, "D4", scale_factor=1.0) 

        # Border for pulsing animation in Line 5
        gold_border = Rectangle(
            width=0.4, height=4, 
            color=COLOR_WHITE, stroke_width=3, fill_opacity=0
        ).move_to(gold_rect.get_center())

        # === Animation for Lecture Line 1 ===
        # 'This unit square represents our entire sample space.'
        self.play(
            self.lecture[0].animate.set_color(COLOR_WHITE),
            Create(square_outline),
            Write(label_sample)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 'We divide it to represent different possible outcomes.'
        self.play(
            self.lecture[1].animate.set_color(COLOR_WHITE),
            Create(divider)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # 'Ten percent represent rare Golden Squirrels, others are Grey.'
        self.play(
            self.lecture[2].animate.set_color(COLOR_GOLD),
            FadeIn(gold_rect),
            FadeIn(grey_rect)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # 'These areas show the initial probability of each group.'
        self.play(
            self.lecture[3].animate.set_color(COLOR_GREY),
            Write(label_gold),
            Write(label_grey)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # 'This gold strip is our prior belief for Golden Squirrels.'
        self.play(
            self.lecture[4].animate.set_color(COLOR_GOLD),
            Create(gold_border)
        )
        self.play(
            Indicate(gold_border, color=COLOR_WHITE, scale_factor=1.1),
            run_time=2
        )
        self.wait(2)
