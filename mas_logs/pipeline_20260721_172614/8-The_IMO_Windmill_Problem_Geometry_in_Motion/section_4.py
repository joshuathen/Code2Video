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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetching storyboard data
        title_str = "The 180-Degree Flip"
        lecture_lines = [
            "Watch what happens after a full 180-degree rotation.",
            "The line's orientation has completely flipped its direction.",
            "Points once on the left are now on the right.",
            "For this swap, the line must hit every point.",
            "This proves the windmill visits every point eventually."
        ]
        
        # Format with bullets for layout
        bullet_lines = ["- " + line for line in lecture_lines]
        self.setup_layout(title_str, bullet_lines)

        # Colors
        COLOR_LINE = "#FFFF00"
        COLOR_PIVOT = "#2ECC71"
        COLOR_RED = "#E74C3C"
        COLOR_BLUE = "#3498DB"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_LINE))
        
        # Setup Grid Elements
        pivot_dot = Dot(color=COLOR_PIVOT)
        self.place_at_grid(pivot_dot, "C4")
        
        red_pts = VGroup(
            Dot(color=COLOR_RED),
            Dot(color=COLOR_RED)
        )
        self.place_at_grid(red_pts[0], "B3")
        self.place_at_grid(red_pts[1], "A3")
        
        blue_pts = VGroup(
            Dot(color=COLOR_BLUE),
            Dot(color=COLOR_BLUE)
        )
        self.place_at_grid(blue_pts[0], "D5")
        self.place_at_grid(blue_pts[1], "E5")
        
        all_dots = VGroup(pivot_dot, *red_pts, *blue_pts)
        self.add(all_dots)

        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg]
        # We use the SVG as the visual for the rotating line
        windmill_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        windmill_asset.set_color(COLOR_LINE)
        windmill_asset.scale_to_fit_height(5.0)
        windmill_asset.move_to(pivot_dot.get_center())
        
        # Arrow pointing 'North'
        arrow = Arrow(
            start=pivot_dot.get_center(),
            end=pivot_dot.get_center() + UP * 1.5,
            buff=0,
            color=WHITE,
            stroke_width=6,
            max_tip_length_to_length_ratio=0.25
        )
        
        windmill = VGroup(windmill_asset, arrow)
        self.add(windmill)

        # Full 180-degree rotation clockwise
        self.play(Rotate(windmill, angle=-PI, about_point=pivot_dot.get_center()), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(Indicate(arrow, color=WHITE, scale_factor=1.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_BLUE))
        
        # Labels for line-relative sides
        label_line_left = Text("Line's Left", font_size=16, color=WHITE)
        label_line_right = Text("Line's Right", font_size=16, color=WHITE)
        
        # VideoCritic Fixes (Issue 33, 34)
        # When arrow is DOWN, Line's Left is Screen-Right (C5), Line's Right is Screen-Left (C3)
        self.place_at_grid(label_line_left, 'C5', scale_factor=0.8)
        self.place_at_grid(label_line_right, 'C3', scale_factor=0.8)
        
        self.play(FadeIn(label_line_left), FadeIn(label_line_right))
        self.play(Indicate(blue_pts, color=COLOR_BLUE), Indicate(red_pts, color=COLOR_RED))
        self.wait(2)
        self.play(FadeOut(label_line_left), FadeOut(label_line_right))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_LINE))
        
        # Reset rotation for segment display to show point hits
        self.play(Rotate(windmill, angle=PI, about_point=pivot_dot.get_center()), run_time=1)
        
        # Rotation in segments with flashes as the line hits each point
        # Angles calculated based on grid positions relative to C4
        # A3: -26.5, B3: -45, D5: 135, E5: 153.4 degrees from vertical North
        self.play(Rotate(windmill, angle=-26.5*DEGREES, about_point=pivot_dot.get_center()), run_time=0.8)
        self.play(Indicate(red_pts[1], color=COLOR_LINE))
        
        self.play(Rotate(windmill, angle=-(45-26.5)*DEGREES, about_point=pivot_dot.get_center()), run_time=0.5)
        self.play(Indicate(red_pts[0], color=COLOR_LINE))
        
        self.play(Rotate(windmill, angle=-(135-45)*DEGREES, about_point=pivot_dot.get_center()), run_time=1.2)
        self.play(Indicate(blue_pts[0], color=COLOR_LINE))
        
        self.play(Rotate(windmill, angle=-(153.4-135)*DEGREES, about_point=pivot_dot.get_center()), run_time=0.5)
        self.play(Indicate(blue_pts[1], color=COLOR_LINE))
        
        self.play(Rotate(windmill, angle=-(180-153.4)*DEGREES, about_point=pivot_dot.get_center()), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_PIVOT))
        self.play(all_dots.animate.set_color(COLOR_PIVOT))
        self.wait(3)
