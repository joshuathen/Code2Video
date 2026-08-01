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
        # Setup layout
        lecture_lines = [
            'Shifting dimensions lets us fold space itself.',
            'Folding a ribbon brings distant points together.',
            'Higher dimensions create shortcuts impossible in 2D.'
        ]
        self.setup_layout("The Wormhole Shortcut: Solving Dimensional Puzzles", lecture_lines)

        # Colors
        RIBBON_COLOR = "#DEB887"
        CAT_COLOR = WHITE
        MOUSE_COLOR = GRAY
        LINE1_COLOR = YELLOW
        LINE2_COLOR = TEAL
        LINE3_COLOR = GREEN

        # === Animation for Lecture Line 1 ===
        # Use color to highlight current line
        self.play(self.lecture[0].animate.set_color(LINE1_COLOR))
        
        # Initial Straight Ribbon
        ribbon = VMobject()
        ribbon.set_points_as_corners([self.grid["C1"], self.grid["C5"]])
        ribbon.set_stroke(color=RIBBON_COLOR, width=20)
        
        cat = Dot(color=CAT_COLOR, radius=0.15)
        self.place_at_grid(cat, "C1")
        cat_label = Text("Cat", font_size=20, color=CAT_COLOR)
        self.place_at_grid(cat_label, "B1")
        
        mouse = Dot(color=MOUSE_COLOR, radius=0.15)
        self.place_at_grid(mouse, "C5")
        mouse_label = Text("Mouse", font_size=20, color=MOUSE_COLOR)
        # Fix for Issue 41: Position mouse_label to avoid convergence overlap
        self.place_at_grid(mouse_label, "D5")

        self.play(Create(ribbon))
        self.play(FadeIn(cat, cat_label, mouse, mouse_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset previous line, highlight current
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(LINE2_COLOR)
        )
        
        # Bend ribbon into a 'U' shape
        u_ribbon = VMobject()
        u_ribbon.set_points_as_corners([
            self.grid["C1"],
            self.grid["E1"],
            self.grid["E5"],
            self.grid["C5"]
        ]).make_smooth()
        u_ribbon.set_stroke(color=RIBBON_COLOR, width=20)
        
        self.play(
            Transform(ribbon, u_ribbon),
            run_time=2
        )
        
        # Visualize the distance along the surface vs. spatial distance
        dist_indicator = DashedLine(self.grid["C1"], self.grid["C5"], color=WHITE)
        dist_text = Text("Distant", font_size=16, color=WHITE)
        # Fix for Issue 40: Place label in area that doesn't overlap the dashed line
        self.place_in_area(dist_text, "B2", "B4", scale_factor=0.8)
        
        self.play(Create(dist_indicator), FadeIn(dist_text))
        self.wait(1)
        self.play(FadeOut(dist_indicator, dist_text))

        # === Animation for Lecture Line 3 ===
        # Reset previous line, highlight current
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(LINE3_COLOR)
        )
        
        # Fold ribbon ends together (Shortcut)
        # We define a "loop" where endpoints meet at C3
        folded_ribbon = VMobject()
        folded_ribbon.set_points_as_corners([
            self.grid["C3"],
            self.grid["F1"],
            self.grid["F5"],
            self.grid["C3"]
        ]).make_smooth()
        folded_ribbon.set_stroke(color=RIBBON_COLOR, width=20)
        
        # Move objects to meet at C3
        self.play(
            Transform(ribbon, folded_ribbon),
            cat.animate.move_to(self.grid["C3"]),
            mouse.animate.move_to(self.grid["C3"]),
            cat_label.animate.move_to(self.grid["B3"]),
            mouse_label.animate.move_to(self.grid["D3"]),
            run_time=2
        )
        
        # Final visual feedback of the shortcut
        shortcut_flash = Flash(self.grid["C3"], color=WHITE, flash_radius=0.5)
        success_glow = Dot(self.grid["C3"], radius=0.3, color=LINE3_COLOR).set_opacity(0.3)
        self.play(FadeIn(success_glow), shortcut_flash)
        
        self.wait(3)
