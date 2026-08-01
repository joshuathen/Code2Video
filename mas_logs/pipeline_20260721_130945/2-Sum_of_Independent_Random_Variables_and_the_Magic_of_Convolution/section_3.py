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

class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Discrete Case: The Grid Method"
        lecture_lines = [
            "Imagine rolling two dice and summing the results.",
            "A grid shows every possible pair of outcomes.",
            "Sum probabilities along the diagonal for each total."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_DICE = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFA500"
        
        # === Animation for Lecture Line 1 ===
        # "Imagine rolling two dice and summing the results."
        self.lecture[0].set_color(COLOR_DICE)
        
        # Load dice asset
        dice_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg"
        die1 = SVGMobject(dice_asset_path).set_color(COLOR_DICE)
        die2 = SVGMobject(dice_asset_path).set_color(COLOR_DICE)
        dice_group = VGroup(die1, die2).arrange(RIGHT, buff=0.5)
        
        # Fix for Issue 32: Place at A4, scale 1.0 (prevents overlap with grid)
        self.place_at_grid(dice_group, 'A4', scale_factor=1.0)
        
        self.play(FadeIn(dice_group))
        
        # Simulate rolling (simple rotation and scale pulses)
        for _ in range(3):
            self.play(
                dice_group.animate.rotate(PI/2).scale(1.1),
                run_time=0.2
            )
            self.play(
                dice_group.animate.scale(1/1.1),
                run_time=0.2
            )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "A grid shows every possible pair of outcomes."
        self.lecture[1].set_color(WHITE)
        
        # Create 6x6 grid
        grid_squares = VGroup()
        square_size = 0.5
        for r in range(6):
            for c in range(6):
                sq = Square(side_length=square_size, color=GREY_A).set_stroke(width=1)
                sq.move_to(np.array([c * square_size, -r * square_size, 0]))
                grid_squares.add(sq)
        
        # Labels for X and Y outcomes
        x_labels = VGroup(*[Text(str(i+1), font_size=16) for i in range(6)])
        y_labels = VGroup(*[Text(str(i+1), font_size=16) for i in range(6)])
        
        # Initial grid positioning before place_in_area to align labels correctly
        grid_with_labels = VGroup(grid_squares)
        
        # Position x labels above top row
        for i in range(6):
            x_labels[i].next_to(grid_squares[i], UP, buff=0.1)
        grid_with_labels.add(x_labels)
        
        # Position y labels left of first column
        for i in range(6):
            y_labels[i].next_to(grid_squares[i*6], LEFT, buff=0.1)
        grid_with_labels.add(y_labels)

        # Fix for Issue 33: Use area B2-F6, scale factor 0.9 (provides enough room for labels)
        self.place_in_area(grid_with_labels, 'B2', 'F6', scale_factor=0.9)
        
        self.play(
            FadeOut(dice_group),
            Create(grid_squares),
            Write(x_labels),
            Write(y_labels),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Sum probabilities along the diagonal for each total."
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Highlight diagonal where x + y = 4 (sum is 4)
        # Outcomes: (1,3), (2,2), (3,1)
        # Indices in 6x6: Row 0 Col 2 (2), Row 1 Col 1 (7), Row 2 Col 0 (12)
        target_indices = [2, 7, 12]
        highlights = VGroup()
        pair_texts = VGroup()
        
        for idx in target_indices:
            h = grid_squares[idx].copy().set_fill(COLOR_HIGHLIGHT, opacity=0.5).set_stroke(color=COLOR_HIGHLIGHT, width=2)
            highlights.add(h)
            
            # Label the cell with its sum components
            r = idx // 6
            c = idx % 6
            txt = Text(f"{r+1},{c+1}", font_size=12).move_to(grid_squares[idx].get_center())
            pair_texts.add(txt)
            
        self.play(
            LaggedStart(
                FadeIn(highlights),
                Write(pair_texts),
                lag_ratio=0.5
            )
        )
        
        # Show the diagonal line
        # Line from top-right corner of (1,3) to bottom-left corner of (3,1)
        diag_line = Line(
            grid_squares[2].get_corner(UP+RIGHT),
            grid_squares[12].get_corner(DOWN+LEFT),
            color=COLOR_HIGHLIGHT,
            stroke_width=4
        )
        
        self.play(Create(diag_line))
        self.play(Indicate(highlights, scale_factor=1.1, color=COLOR_HIGHLIGHT))
        self.wait(2)
