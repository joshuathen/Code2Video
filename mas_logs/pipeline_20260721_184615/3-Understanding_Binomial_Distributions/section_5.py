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

class Section5Scene(TeachingScene):
    def construct(self):
        # Fetching current shared state info for consistency
        title_text = "Application and Summary"
        lecture_lines = [
            "Binomial distributions help us predict real-world outcomes.",
            "Quality control uses this to find defect probabilities.",
            "It's a powerful tool for repeated, independent trials."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors for highlights and icons
        COLOR_LINE_1 = "#00FFFF"
        COLOR_LINE_2 = "#FFFF00"
        COLOR_LINE_3 = "#00FF00"
        COLOR_ROBOT = "#C0C0C0"
        COLOR_DEFECT = "#FF0000"
        COLOR_BOARD = "#2E7D32"  # Dark green for circuit board
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # [self.wait(2.0)] Fade in Robot icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] (#C0C0C0) and 100 circuit boards.
        self.play(self.lecture[0].animate.set_color(COLOR_LINE_1))
        
        # Asset integration (Issue 19)
        try:
            robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
            robot.set_color(COLOR_ROBOT)
        except Exception:
            # Fallback robot construction
            robot_head = Square(side_length=0.4, color=COLOR_ROBOT, fill_opacity=1)
            robot_body = Square(side_length=0.7, color=COLOR_ROBOT, fill_opacity=1).next_to(robot_head, DOWN, buff=0.1)
            robot = VGroup(robot_head, robot_body)
        
        # Fix (Issue 27): Horizontal alignment and vertical crowding
        self.place_at_grid(robot, "B4", scale_factor=0.8)

        # 100 Circuit Boards (10x10 grid)
        boards = VGroup()
        for i in range(100):
            board = Rectangle(width=0.3, height=0.2, color=COLOR_BOARD, fill_opacity=0.8, stroke_width=1)
            boards.add(board)
        
        boards.arrange_in_grid(rows=10, cols=10, buff=0.1)
        # Fix (Issue 28): Prevent overlap with robot by moving to area starting at C3
        self.place_in_area(boards, "C3", "F6", scale_factor=0.7)

        self.play(FadeIn(robot), FadeIn(boards))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # [self.wait(1.5)] Highlight defective circuit boards in Red (#FF0000).
        self.play(self.lecture[1].animate.set_color(COLOR_LINE_2))
        
        # Selecting a few indices to represent defects (low probability demo)
        defect_indices = [12, 37, 58, 81]
        defects = VGroup(*[boards[i] for i in defect_indices])
        
        self.play(
            *[Indicate(board, color=COLOR_DEFECT) for board in defects],
            *[board.animate.set_fill(COLOR_DEFECT).set_stroke(COLOR_DEFECT) for board in defects]
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # [self.wait(2.0)] Fade out icons; show summary text in #FFFFFF.
        self.play(self.lecture[2].animate.set_color(COLOR_LINE_3))
        
        self.play(FadeOut(robot), FadeOut(boards))
        self.wait(0.5)

        # Summary text
        summary_title = Text("Key Properties", font_size=26, color=COLOR_TEXT, weight=BOLD)
        summary_points = VGroup(
            Text("• Predicts outcomes in repeated trials", font_size=20, color=COLOR_TEXT),
            Text("• Assumes independence between trials", font_size=20, color=COLOR_TEXT),
            Text("• Constant probability of success (p)", font_size=20, color=COLOR_TEXT),
            Text("• Applicable to quality control & science", font_size=20, color=COLOR_TEXT)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        summary_vgroup = VGroup(summary_title, summary_points).arrange(DOWN, buff=0.5)
        # Fix (Issue 29): Properly scale and center summary text
        self.place_in_area(summary_vgroup, "B2", "E5", scale_factor=0.8)

        self.play(FadeIn(summary_vgroup))
        self.wait(2.0)
