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

class Section1Scene(TeachingScene):
    def construct(self):
        title_str = "Introduction: The Blurry Vision Problem"
        lecture_lines = [
            "Convolution is the math behind digital photo filters.",
            "It combines two data sets to create a third.",
            "We'll use it to clear a robot's blurry vision."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        GRAY_COLOR = "#808080"
        BRUSH_COLOR = "#FFFF00"
        HIGHLIGHT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a 6x6 grid of gray squares
        grid_squares = VGroup()
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]
        
        for r in rows:
            for c in cols:
                # Add some "noise" via varying initial opacities
                op = np.random.uniform(0.1, 0.4)
                sq = Square(side_length=0.9, fill_color=GRAY_COLOR, fill_opacity=op, stroke_width=1, stroke_color=GRAY_COLOR)
                self.place_at_grid(sq, f"{r}{c}")
                grid_squares.add(sq)
        
        self.play(FadeIn(grid_squares))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Introduce a 'Mathematical Brush' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/brush.svg]
        # Using SVGMobject and applying scale factor of 0.8 as per Issue 40
        brush = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brush.svg")
        brush.set_color(BRUSH_COLOR)
        self.place_in_area(brush, 'A1', 'B2', scale_factor=0.8)
        
        brush_label = Text("Brush", font_size=18, color=BRUSH_COLOR)
        # Position label relative to brush center to avoid title overlap
        brush_label.next_to(brush, UP, buff=0.1)
        
        self.play(DrawBorderThenFill(brush), Write(brush_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # The brush sweeps over the grid, turning it into a clear 'L' shape.
        # Define 'L' indices (0-35): B2, C2, D2, E2, E3, E4, E5
        l_indices = [7, 13, 19, 25, 26, 27, 28]
        
        # Tracker for the brush movement to trigger color changes
        def update_grid(mobject):
            brush_center = brush.get_center()
            for i, sq in enumerate(grid_squares):
                # If brush is close to square, reveal if it's part of L, else clear noise
                dist = np.linalg.norm(sq.get_center() - brush_center)
                if dist < 1.1:
                    if i in l_indices:
                        sq.set_fill(HIGHLIGHT_COLOR, opacity=0.9)
                        sq.set_stroke(HIGHLIGHT_COLOR, width=2)
                    else:
                        sq.set_fill(GRAY_COLOR, opacity=0.05)
                        sq.set_stroke(GRAY_COLOR, width=1)

        grid_squares.add_updater(update_grid)
        
        # Define the sweep path
        sweep_positions = [
            ('A5', 'B6'), # Sweep top rows
            ('C1', 'D2'), # Middle rows
            ('C5', 'D6'),
            ('E1', 'F2'), # Bottom rows
            ('E5', 'F6')
        ]
        
        for pos_start, pos_end in sweep_positions:
            # Calculate target center for the brush
            tl_pos = self.grid[pos_start]
            br_pos = self.grid[pos_end]
            target_center = (tl_pos + br_pos) / 2
            
            self.play(
                brush.animate.move_to(target_center),
                brush_label.animate.next_to(target_center, UP, buff=0.1),
                run_time=0.7,
                rate_func=linear
            )

        grid_squares.remove_updater(update_grid)
        self.play(FadeOut(brush), FadeOut(brush_label))
        self.wait(2)
